from typing import Any


def crop_frame(
    frame,
    crop_config
) -> Any:
    '''Вырезание области по координатам'''

    x = crop_config.x
    y = crop_config.y

    if x < 0 or y < 0:
        raise ValueError('Параметры x, y должны быть >= 0')

    width = crop_config.width
    height = crop_config.height

    if width <= 0 or height <= 0:
        raise ValueError('Параметры width, height должны быть > 0')
    
    frame_height, frame_width = frame.shape[:2]
    if x + width > frame_width or y + height > frame_height:
        raise ValueError(
            'OCR crop выходит за границы кадра: '
            f'crop=({x}, {y}, {width}, {height}), '
            f'frame=({frame_width}, {frame_height})'
        )

    return frame[y:y + height, x:x + width]