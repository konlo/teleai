"""Validate locally generated chart bytes before publishing or displaying them."""
from io import BytesIO
import warnings
from PIL import Image, UnidentifiedImageError


def validate_chart_image(payload):
    if not isinstance(payload, bytes) or len(payload) > 16 * 1024 * 1024:
        raise ValueError('차트 이미지 데이터가 없거나 크기 한도를 초과했습니다.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format != 'PNG' or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('차트는 단일 PNG 이미지여야 합니다.')
                width, height = image.size
                if min(width, height) < 64 or width * height > 16_000_000:
                    raise ValueError('차트 이미지의 표시 크기가 유효하지 않습니다.')
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                image.load()
                # Composite transparency onto the same light background used by the UI.
                rgba = image.convert('RGBA')
                visible = Image.new('RGBA', rgba.size, 'white')
                visible.alpha_composite(rgba)
                if all(high == low for low, high in visible.convert('RGB').getextrema()):
                    raise ValueError('차트 이미지가 비어 있습니다.')
        return {'width': width, 'height': height}
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('차트 이미지가 손상되어 표시할 수 없습니다.') from exc
