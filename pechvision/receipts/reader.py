from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
from pandas import DataFrame


def read_receipts_file(path: str | Path) -> DataFrame:
    '''Загрузка файла чеков (.xlsx / .csv)'''

    file_format = Path(path).suffix.lower()

    if file_format == '.csv':
        return pd.read_csv(path)
    
    if file_format == '.xlsx':
        return pd.read_excel(path)
    
    raise ValueError(f'Загружен неподдерживаемый формат файла чеков: {file_format}')


def normalize_amount_csv(value: str | None) -> Decimal | None:
    '''Хелпер-функция нормализации суммы чека из формата .csv'''

    if not value:
        return None
    
    return Decimal(
        value.replace('₽', '').replace(' ', '').replace(',', '.').strip()
    ).quantize(
        Decimal('0.00'),
        rounding=ROUND_HALF_UP
    )


def normalize_receipt_datetime(time: str | None) -> datetime | None:
    '''Хелпер-функция номрализации времени открытия/закрытия чека'''

    if not time:
        return None
    return datetime.fromisoformat(time)


def normalize_receipt_row(row: dict, source_file: str | Path) -> dict:
    '''Нормализация строки для таблицы receipts'''

    normalize_row = {
        'external_receipt_id': str(row['OrderId']) if pd.notna(row['OrderId']) else None,
        'tt': str(row['TT']) if pd.notna(row['TT']) else None,
        'opened_at': (
            normalize_receipt_datetime(row['openTime'])
            if pd.notna(row['openTime'])
            else None
        ),
        'closed_at': (
            normalize_receipt_datetime(row['dl_tm'])
            if pd.notna(row['dl_tm'])
            else None
        ),
        'amount': (
            normalize_amount_csv(row['OrderSum'])
            if pd.notna(row['OrderSum'])
            else None
        ),
        'table_number': int(row['stol_num']) if pd.notna(row['stol_num']) else None,
        'client_external_id': str(row['Client_TTGID']) if pd.notna(row['Client_TTGID']) else None,
        'source_file': str(source_file),
        'raw_data': {
            'TT': str(row['TT']),
            'OrderId': str(row['OrderId']),
            'openTime': str(row['openTime']),
            'dl_tm': str(row['dl_tm']),
            'OrderSum': str(row['OrderSum']),
            'stol_num': str(row['stol_num']),
            'Client_TTGID': str(row['Client_TTGID'])
        }
    }
    return normalize_row


def normalize_receipts_file(path: str | Path) -> list[dict]:
    '''Создание списка нормализованных словарей строк входного файла'''

    if not path:
        raise RuntimeError('Параметр path не передан или передан с ошибкой')

    df = read_receipts_file(path)
    rows = df.to_dict('records')

    return [
        normalize_receipt_row(row, source_file=path)
        for row in rows
    ] 