#!/usr/bin/env python3
import os
from datetime import datetime

import spacetrack.operators as op
from dotenv import load_dotenv
from sgp4.api import Satrec
# Главный класс для работы с space-track
from spacetrack import SpaceTrackClient

# Имя пользователя и пароль сейчас опишем как константы

load_dotenv()

# Имя пользователя и пароль сейчас опишем как константы
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

utc_time = datetime.utcnow()


def get_spacetrack_tle(
    sat_id, start_date, end_date, username, password, latest=False
):
    # Реализуем экземпляр класса SpaceTrackClient
    st = SpaceTrackClient(identity=username, password=password)
    # Выполнение запроса для диапазона дат:
    if not latest:
        # Определяем диапазон дат через оператор библиотеки
        daterange = op.inclusive_range(start_date, end_date)
        # Собственно выполняем запрос через st.tle
        data = st.tle(
            norad_cat_id=sat_id,
            orderby="epoch desc",
            limit=1,
            format="tle",
            epoch=daterange,
        )
    # Выполнение запроса для актуального состояния
    else:
        # Выполняем запрос через st.tle_latest
        data = st.tle_latest(
            norad_cat_id=sat_id, orderby="epoch desc", limit=1, format="tle"
        )

    # Если данные недоступны
    if not data:
        return 0, 0

    # Иначе возвращаем две строки
    tle_1 = data[0:69]
    tle_2 = data[70:139]
    return tle_1, tle_2


def read_tle_base_file(norad_number):
    with open("1.tle", "r") as fp:
        lines = fp.readlines()

    sats = []

    for i in range(len(lines) - 1):
        if (lines[i][0] == "1") and (lines[i+1][0] == "2"):
            sats.append(lines[i-1] + lines[i] + lines[i+1])

    for j in range(len(sats)):
        tle_string = sats[j]
        s_name, tle_1, tle_2 = tle_string.strip().splitlines()
        sat = Satrec.twoline2rv(tle_1, tle_2)
        if sat.satnum == norad_number:
            return s_name, tle_1, tle_2

def read_tle_base_internet(norad_number):
    tle_i_1, tle_i_2 = get_spacetrack_tle(norad_number, None, None, USERNAME, PASSWORD, True)
    s_name_f, tle_f_1, tle_f_2 = read_tle_base_file(norad_number)
    return s_name_f, tle_i_1, tle_i_2

def _test():
    print(read_tle_base_file(56756))
#    print(read_tle_base_internet(56756))


if __name__ == "__main__":
    _test()
