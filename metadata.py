import xml.etree.ElementTree as ET
from datetime import datetime

# Функция для создания XML-метаданных
def create_metadata(output_path, radar_data_info):
    # Создание корневого элемента
    root = ET.Element("RadarMetadata")

    # Добавление общей информации
    mission = ET.SubElement(root, "Mission")
    mission.text = "Кондор-ФКА"

    mode = ET.SubElement(root, "Mode")
    mode.text = "Прожекторный режим"

    # Добавление информации о данных
    data_info = ET.SubElement(root, "DataInfo")
    
    resolution = ET.SubElement(data_info, "Resolution")
    resolution.text = f"{radar_data_info['resolution']} м"

    width = ET.SubElement(data_info, "Width")
    width.text = f"{radar_data_info['width']} км"

    height = ET.SubElement(data_info, "Height")
    height.text = f"{radar_data_info['height']} км"

    acquisition_date = ET.SubElement(data_info, "AcquisitionDate")
    acquisition_date.text = radar_data_info['acquisition_date']

    # Добавление информации о местоположении
    location = ET.SubElement(root, "Location")
    
    latitude = ET.SubElement(location, "Latitude")
    latitude.text = str(radar_data_info['latitude'])

    longitude = ET.SubElement(location, "Longitude")
    longitude.text = str(radar_data_info['longitude'])

    # Сохранение XML-файла
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"Метаданные сохранены в файл: {output_path}")

# Основная программа
if __name__ == "__main__":
    # Пример данных для метаданных
    radar_data_info = {
        "resolution": 1.0,  # Разрешение в метрах
        "width": 5.0,       # Ширина кадра в км
        "height": 10.0,     # Длина кадра в км
        "acquisition_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Дата и время съемки
        "latitude": 55.7558,  # Широта центра кадра
        "longitude": 37.6173  # Долгота центра кадра
    }

    # Путь для сохранения метаданных
    output_path = "condor_fka_spotlight_metadata.xml"

    # Создание метаданных
    create_metadata(output_path, radar_data_info)