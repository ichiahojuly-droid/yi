import math
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from pathlib import Path

WORKBOOK_PATH = Path(__file__).resolve().parent.parent / '全省 每小时平均.xlsx'


def read_excel_to_dicts(xlsx_path):
    with ZipFile(xlsx_path) as zf:
        shared_strings = []
        if 'xl/sharedStrings.xml' in zf.namelist():
            with zf.open('xl/sharedStrings.xml') as fh:
                root = ET.fromstring(fh.read())
                for si in root:
                    text = ''.join(node.text or '' for node in si.iter() if node.tag.endswith('}t'))
                    shared_strings.append(text)
        with zf.open('xl/worksheets/sheet1.xml') as fh:
            sheet_root = ET.fromstring(fh.read())
    rows = []
    for row in sheet_root.findall('.//{*}row'):
        row_index = int(row.attrib['r'])
        row_data = {}
        for cell in row.findall('{*}c'):
            ref = cell.attrib.get('r')
            match = re.match(r'([A-Z]+)(\d+)', ref or '')
            if not match:
                continue
            col, _ = match.groups()
            value_node = cell.find('{*}v')
            if value_node is None:
                continue
            value = value_node.text
            cell_type = cell.attrib.get('t')
            if cell_type == 's':
                value = shared_strings[int(value)]
            else:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            row_data[col] = value
        rows.append((row_index, row_data))
    rows.sort(key=lambda x: x[0])
    header = rows[0][1]
    col_map = {col: header[col] for col in header}
    data = []
    for _, row_data in rows[1:]:
        record = {name: row_data.get(col) for col, name in col_map.items()}
        data.append(record)
    return data


def matmul(A, B):
    result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for k in range(len(B)):
            aik = A[i][k]
            for j in range(len(B[0])):
                result[i][j] += aik * B[k][j]
    return result


def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def gauss_jordan_inverse(matrix):
    n = len(matrix)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if abs(aug[r][col]) > 1e-9:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError('Matrix is singular')
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-9:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def linear_regression(records):
    X = [[1.0, r[0], r[1]] for r in records]
    y = [[r[2]] for r in records]
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    XtX_inv = gauss_jordan_inverse(XtX)
    XtY = matmul(Xt, y)
    beta = [b[0] for b in matmul(XtX_inv, XtY)]
    y_mean = sum(r[2] for r in records) / len(records)
    ss_tot = sum((r[2] - y_mean) ** 2 for r in records)
    ss_res = 0.0
    for ws, rad, actual in records:
        pred = beta[0] + beta[1] * ws + beta[2] * rad
        ss_res += (actual - pred) ** 2
    r2 = 1 - ss_res / ss_tot if ss_tot else float('nan')
    return beta, r2


def pearson(x_values, y_values):
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    var_x = sum((x - mean_x) ** 2 for x in x_values)
    var_y = sum((y - mean_y) ** 2 for y in y_values)
    return cov / math.sqrt(var_x * var_y)


def describe(values):
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return mean_val, math.sqrt(variance)


def main():
    data = read_excel_to_dicts(WORKBOOK_PATH)
    wind_key = '100m风速'
    radiation_key = '辐射量'
    wind_power_key = '实时风电(MW)'
    solar_power_key = '实时光伏(MW)'
    temp_key = '温度'
    humidity_key = '湿度'
    load_key = '实时负荷率(%)'

    wind_records = []
    solar_records = []
    for row in data:
        ws = row.get(wind_key)
        rad = row.get(radiation_key)
        wind = row.get(wind_power_key)
        solar = row.get(solar_power_key)
        if ws is not None and rad is not None and wind is not None:
            wind_records.append((ws, rad, wind))
        if ws is not None and rad is not None and solar is not None:
            solar_records.append((ws, rad, solar))

    wind_beta, wind_r2 = linear_regression(wind_records)
    solar_beta, solar_r2 = linear_regression(solar_records)

    stats = {}
    for name, values in (
        ('wind_speed', [ws for ws, _, _ in wind_records]),
        ('radiation', [rad for _, rad, _ in wind_records]),
        ('wind_power', [wp for *_, wp in wind_records]),
        ('solar_power', [sp for *_, sp in solar_records]),
    ):
        mean_val, std_val = describe(values)
        stats[name] = {'mean': mean_val, 'std': std_val}

    correlations = {
        'wind_speed__wind_power': pearson([ws for ws, _, _ in wind_records], [wp for *_, wp in wind_records]),
        'radiation__wind_power': pearson([rad for _, rad, _ in wind_records], [wp for *_, wp in wind_records]),
        'radiation__solar_power': pearson([rad for _, rad, _ in solar_records], [sp for *_, sp in solar_records]),
        'wind_speed__solar_power': pearson([ws for ws, _, _ in solar_records], [sp for *_, sp in solar_records]),
    }

    humidity_values = []
    temperature_values = []
    load_values = []
    for row in data:
        humidity = row.get(humidity_key)
        temperature = row.get(temp_key)
        load = row.get(load_key)
        if humidity is not None and load is not None:
            humidity_values.append(humidity)
            load_values.append(load)
        if temperature is not None and load is not None:
            temperature_values.append(temperature)
    humidity_corr = pearson(humidity_values, load_values)
    temperature_corr = pearson(temperature_values, load_values)

    report = {
        'records': len(data),
        'wind_records': len(wind_records),
        'solar_records': len(solar_records),
        'wind_model': {
            'coefficients': {
                'intercept': wind_beta[0],
                'wind_speed': wind_beta[1],
                'radiation': wind_beta[2],
            },
            'r_squared': wind_r2,
        },
        'solar_model': {
            'coefficients': {
                'intercept': solar_beta[0],
                'wind_speed': solar_beta[1],
                'radiation': solar_beta[2],
            },
            'r_squared': solar_r2,
        },
        'feature_stats': stats,
        'feature_correlations': correlations,
        'load_correlations': {
            'humidity_load': humidity_corr,
            'temperature_load': temperature_corr,
        },
    }

    output_path = Path(__file__).resolve().parent / 'results.json'
    import json
    with output_path.open('w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
