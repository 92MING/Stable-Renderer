'''
A modified module from https://github.com/robertkist/py_dds by Robert Kist
'''
__author__ = "Robert Kist"

import struct
import math
import OpenGL.GL as gl

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union, Type, Tuple, Literal
from OpenGL.GL.EXT.texture_compression_s3tc import *
from OpenGL.GL.EXT.texture_sRGB import *



class DDSException(Exception):
    """a class for throwing DDSImage specific exceptions"""


class DDSFormat(Enum):
    """supported DDS formats, including DX9 and DX10 formats"""
    
    DXT1 = "DXT1"  # includes DXT1a, DXT1c
    DXT3 = "DXT3"  # includes DXT2, DXT3
    DXT5 = "DXT5"  # includes DXT3, DXT4
    UNCOMPRESSED32 = "UNCOMPRESSED32"
    UNCOMPRESSED24 = "UNCOMPRESSED24"
    DXGI_FORMAT_BC1_UNORM_SRGB = "DXGI_FORMAT_BC1_UNORM_SRGB"
    DXGI_FORMAT_BC2_UNORM_SRGB = "DXGI_FORMAT_BC2_UNORM_SRGB"
    DXGI_FORMAT_BC3_UNORM_SRGB = "DXGI_FORMAT_BC3_UNORM_SRGB"
    
    def getOpenGLformat(self):
        """returns the OpenGL texture format for the DDSFormat"""
        if self == DDSFormat.DXT1:
            return GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
        elif self == DDSFormat.DXT3:
            return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
        elif self == DDSFormat.DXT5:
            return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
        elif self == DDSFormat.UNCOMPRESSED32:
            return gl.GL_RGBA
        elif self == DDSFormat.UNCOMPRESSED24:
            return gl.GL_RGB
        elif self == DDSFormat.DXGI_FORMAT_BC1_UNORM_SRGB:
            return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
        elif self == DDSFormat.DXGI_FORMAT_BC2_UNORM_SRGB:
            return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
        elif self == DDSFormat.DXGI_FORMAT_BC3_UNORM_SRGB:
            return GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
        
    def rgbFormat(self)->Literal['RGB', 'RGBA']:
        """returns the OpenGL texture format for the DDSFormat"""
        if self == DDSFormat.UNCOMPRESSED24:
            return 'RGB'
        return 'RGBA'
    
    def bytes_per_pixel(self):
        """returns the number of bytes per pixel for the DDSFormat"""
        if self == DDSFormat.DXT1:
            return 0.5
        elif self == DDSFormat.DXT3:
            return 1
        elif self == DDSFormat.DXT5:
            return 1
        elif self == DDSFormat.UNCOMPRESSED32:
            return 4
        elif self == DDSFormat.UNCOMPRESSED24:
            return 3
        elif self == DDSFormat.DXGI_FORMAT_BC1_UNORM_SRGB:
            return 0.5
        elif self == DDSFormat.DXGI_FORMAT_BC2_UNORM_SRGB:
            return 1
        elif self == DDSFormat.DXGI_FORMAT_BC3_UNORM_SRGB:
            return 1

DDS_HEADER_SIZE = 128
DXT10_HEADER_SIZE = 20
SetPixelCallback = Callable[[int, int, int, int, int, int], None]
BCAlphaClass = Union['_BC1Alpha', '_BC2Alpha', '_BC3Alpha']


# helper dict for mapping DX10 header file formats to the unified DDSFormat enum; for internal use
_DxgiFormat: Dict[int, DDSFormat] = {
    72: DDSFormat.DXGI_FORMAT_BC1_UNORM_SRGB,
    75: DDSFormat.DXGI_FORMAT_BC2_UNORM_SRGB,
    78: DDSFormat.DXGI_FORMAT_BC3_UNORM_SRGB,
}


class _DXTCompression(Enum):
    """enum with supported dxt compression types; for internal use"""
    BC1 = "BC1"  # used in DXT1a/c
    BC2 = "BC2"  # used in DXT3/2
    BC3 = "BC3"  # used in DXT5/4


@dataclass(frozen=True)
class _DDSMip:
    """a container for storing mip map information; for internal use"""
    width: int
    height: int
    offset: int  # data start offset in the dds file in bytes
    size: int  # data size in bytes


@dataclass(frozen=True)
class _Color:
    """a container for a single RGBA color value; for internal use by _DDSPalette"""
    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255


class _DDSPalette:
    """a class for generating color palette information for a 4x4 BC1/2/3 compressed pixel chunk"""
    def __init__(self, data: bytes, ofs: int) -> None:
        color0 = struct.unpack_from("<H", data, ofs)[0]
        color1 = struct.unpack_from("<H", data, ofs + 2)[0]
        self.__palette: List[_Color] = []
        self.__load_palette(color0, color1)

    def get(self, index: int) -> _Color:
        """returns the RGBA color for the given palette index"""
        return self.__palette[index]

    def __lerp(self, vmin: int, vmax: int, value: float) -> int:
        """linearly interpolates value between given min and max"""
        return int(round((vmax - vmin) * value + vmin))

    def __lerp_rgb(self, color1: _Color, color2: _Color, value: float) -> _Color:
        """blends two colors together based on a ratio"""
        r: int = self.__lerp(color1.r, color2.r, value)
        g: int = self.__lerp(color1.g, color2.g, value)
        b: int = self.__lerp(color1.b, color2.b, value)
        return _Color(r, g, b)

    def __get_dxt_color(self, color: int) -> _Color:
        """Converts a 16bit DXT RGB (R(5 bits), G(6 bits), B(5 bits)) value into a regular 8bpp RGB value."""
        r = int(round(255.0 / 31.0 * (color >> 11)))
        g = int(round(255.0 / 63.0 * ((color >> 5) & int('111111', 2))))
        b = int(round(255.0 / 31.0 * (color & int('11111', 2))))
        return _Color(r, g, b)

    def __load_palette(self, color0: int, color1: int) -> None:
        """Constructs all 4 palette entries for a chunk's palette from 2 input colors for DXT1/3/5 compression"""
        c_color0: _Color = self.__get_dxt_color(color0)  # read color 1
        c_color1: _Color = self.__get_dxt_color(color1)  # read color 2
        self.__palette = [c_color0, c_color1]
        if color0 > color1:
            self.__palette.append(self.__lerp_rgb(c_color0, c_color1, 0.3333))  # interpolate color 3
            self.__palette.append(self.__lerp_rgb(c_color0, c_color1, 0.6666))  # interpolate color 4
        else:
            self.__palette.append(self.__lerp_rgb(c_color0, c_color1, 0.5))  # interpolate color 3
            self.__palette.append(_Color(0, 0, 0, 0))  # transparent color


class _BC1Alpha:
    """A dummy class for generating alpha information for a 4x4 BC1 compressed pixel chunk"""

    def __init__(self, *args: Any) -> None:
        pass

    def get(self, color: _Color) -> int:
        """returns the input alpha as DXT1 alpha is calculated from the pixel color itself"""
        return color.a


class _BC2Alpha:
    """A class for generating alpha information for a 4x4 BC2 compressed pixel chunk"""

    def __init__(self, data: bytes, ofs: int) -> None:
        """constructor - reads the chunk's alpha data"""
        self.__alphas: int = struct.unpack_from("<Q", data, ofs)[0]

    def get(self, _: Any) -> int:
        """returns the current pixel's alpha and advances the current pixel to the next pixel"""
        alpha: int = (self.__alphas & 0xf) * 0x11  # mask and expand to full 0-255 range
        self.__alphas = self.__alphas >> 4  # shift alpha by 1 pixel
        return alpha


class _BC3Alpha:
    """A class for generating alpha information for a 4x4 BC3 compressed pixel chunk"""

    def __init__(self, data: bytes, ofs: int) -> None:
        """constructor - reads the chunk's alpha data"""
        alpha0 = int.from_bytes(struct.unpack_from("c", data, ofs)[0], "little")
        alpha1 = int.from_bytes(struct.unpack_from("c", data, ofs + 1)[0], "little")
        self.__code_table: List[int] = [alpha0, alpha1]
        if alpha0 > alpha1:
            for j in range(1, 7):
                self.__code_table.append(int(((7 - j) * alpha0 + j * alpha1) / 7.0))
        else:
            for j in range(1, 5):
                self.__code_table.append(int(((5 - j) * alpha0 + j * alpha1) / 5.0))
            self.__code_table.append(0)
            self.__code_table.append(255)
        part1: int = struct.unpack_from("<I", data, ofs + 2)[0]
        part2: int = struct.unpack_from("<H", data, ofs + 6)[0]
        self.__alphas: int = (part2 << 32) | part1

    def get(self, _: Any) -> int:
        """returns the current pixel's alpha and advances the current pixel to the next pixel"""
        alpha: int = self.__code_table[int(self.__alphas & 0b111)]
        self.__alphas = self.__alphas >> 3
        return alpha


class DDSImage:
    """A class for reading compressed and uncompressed .DDS images"""

    def __init__(self, filename: Union[str, Path]) -> None:
        """constructor"""
        self.__data: bytes = b''  # file buffer - holds entire dds file
        self.__format: Optional[DDSFormat] = None  # image format
        self.__color_mask = _Color()  # bit mask for uncompressed pixel data
        self.__mips: List[_DDSMip] = []  # information about mip maps
        self.__is_dx10: bool = False  # helper flag to make it easier to deal with DX10 format
        self.__load(filename)

    def data(self) -> bytes:
        """returns the raw image data"""
        return self.__data

    def width(self, mip: int = 0) -> int:
        """returns the width of the specified mip map number. 0 is the first (largest) mip map"""
        if 0 > mip >= len(self.__mips):
            raise IndexError
        return self.__mips[mip].width

    def height(self, mip: int = 0) -> int:
        """returns the height of the specified mip map number. 0 is the first (largest) mip map"""
        if 0 > mip >= len(self.__mips):
            raise IndexError
        return self.__mips[mip].height

    def format(self) -> Optional[DDSFormat]:
        """returns the DXT image format"""
        return self.__format

    def mip_count(self) -> int:
        """returns the number of mip maps in the file"""
        return len(self.__mips)

    def mip_map(self, mip: int = 0) -> _DDSMip:
        """returns the specified mip map as a bytes object"""
        if 0 > mip >= len(self.__mips):
            raise IndexError
        return self.__mips[mip]

    def compressed(self) -> bool:
        """returns True if the image is compressed, False if it is uncompressed"""
        return self._compressed

    def __set_format(self, pixel_format_flags: int) -> DDSFormat:
        """determines the image format, e.g. DXT1, DXT3, etc. - see DDSFormat enum for supported formats"""
        compressed = bool(pixel_format_flags & 0x4)
        self._compressed = compressed
        if (not compressed and not bool(pixel_format_flags & 0x40)) or (compressed and bool(pixel_format_flags & 0x40)):
            raise DDSException("cannot determine DXT format")
        if compressed:
            four_cc: str = self.__data[84:88].decode("utf8")
            if four_cc == "DX10":
                self.__is_dx10 = True
                try:
                    return _DxgiFormat[struct.unpack_from("<I", self.__data, DDS_HEADER_SIZE)[0]]
                except KeyError as exc:
                    raise DDSException("unsupported DXT10 format") from exc
            try:
                return DDSFormat[four_cc]
            except KeyError as exc:
                raise DDSException("unsupported DXT format") from exc
        bit_count: int = struct.unpack_from("<I", self.__data, 88)[0]
        if bit_count not in [32, 24]:
            raise DDSException("unsupported bit depth")
        return DDSFormat.UNCOMPRESSED32 if bit_count == 32 else DDSFormat.UNCOMPRESSED24

    def __load(self, filename: Union[str, Path]) -> None:
        """loads the DDS file from the given file path"""
        
        with open(filename, 'rb') as fp:
            self.__data = fp.read()
        
        # check if proper DDS file format
        if not (self.__data[0:3] == b"DDS" and self.__data[4] == 124 and self.__data[76] == 32):
            raise DDSException("not a valid DDS image")
        
        # read DDS header values
        dds_flags: int = struct.unpack_from("<I", self.__data, 8)[0]
        height: int = struct.unpack_from("<I", self.__data, 12)[0]
        width: int = struct.unpack_from("<I", self.__data, 16)[0]
        mip_count: int = struct.unpack_from("<I", self.__data, 28)[0]
        pixel_format_flags: int = struct.unpack_from("<I", self.__data, 80)[0]
        
        if bool(dds_flags & 0x800000):
            raise DDSException("cube maps are not supported")
        
        self.__format = self.__set_format(pixel_format_flags)
        self.__mip_count = mip_count if mip_count > 0 else 1
        
        # load mip maps
        if self.__format in [DDSFormat.UNCOMPRESSED32, DDSFormat.UNCOMPRESSED24]:
            self.__get_uncompressed_mip_maps_and_color_info(width, height)
        elif self.__format in [DDSFormat.DXT1, DDSFormat.DXGI_FORMAT_BC1_UNORM_SRGB]:
            self.__get_compressed_mip_maps(width, height, 8)  # 64 bit chunks
        elif self.__format in [DDSFormat.DXT3, DDSFormat.DXGI_FORMAT_BC2_UNORM_SRGB, DDSFormat.DXT5,
                               DDSFormat.DXGI_FORMAT_BC3_UNORM_SRGB]:
            self.__get_compressed_mip_maps(width, height, 16)  # 128 bit chunks

    def __get_uncompressed_mip_maps_and_color_info(self, width: int, height: int) -> None:
        """determines color masks and gets uncompressed mip map info; width and height: dimensions of first mip map"""
        # get color and optional alpha masks
        r: int = struct.unpack_from("<I", self.__data, 92)[0]
        g: int = struct.unpack_from("<I", self.__data, 96)[0]
        b: int = struct.unpack_from("<I", self.__data, 100)[0]
        if self.__format == DDSFormat.UNCOMPRESSED32:
            a: int = struct.unpack_from("<I", self.__data, 104)[0]
            self.__color_mask = _Color(r, g, b, a)
            bytes_per_pixel = 4
        else:
            self.__color_mask = _Color(r, g, b, 255)
            self.__data += b"\x00"  # pad buffer so we can read whole 32-bit words when decoding the image
            bytes_per_pixel = 3
        # load mip maps
        offset: int = DDS_HEADER_SIZE
        for _ in range(0, self.__mip_count):
            self.__mips.append(_DDSMip(width, height, offset, width * height * bytes_per_pixel))
            offset += width * height * bytes_per_pixel
            width = int(width / 2)
            height = int(height / 2)

    def __get_compressed_mip_maps(self, width: int, height: int, chunk_size: int) -> None:
        """gets uncompressed mip map info; width and height: dimensions of first mip map"""
        offset: int = DDS_HEADER_SIZE + DXT10_HEADER_SIZE if self.__is_dx10 else DDS_HEADER_SIZE
        for _ in range(0, self.__mip_count):
            size = math.ceil(width / 4) * math.ceil(height / 4) * chunk_size
            self.__mips.append(_DDSMip(width, height, offset, size))
            offset += size
            width = int(width / 2)
            height = int(height / 2)

    def __draw_compressed(self, set_pixel_callback: SetPixelCallback, data: bytes, width: int, height: int,
                          compression: _DXTCompression) -> None:
        """outputs the given BC1/2/3 compressed mip map via the given set_pixel callback."""
        ofs = x = y = i = 0
        mapper: Dict[_DXTCompression, Tuple[Type[BCAlphaClass], int]] = {
            _DXTCompression.BC1: (_BC1Alpha, 0),  # dxt1 - if alpha is present, 1bit alpha is part of pixel data
            _DXTCompression.BC2: (_BC2Alpha, 8),  # dxt2/3 - has alpha
            _DXTCompression.BC3: (_BC3Alpha, 8),  # dxt4/5 - has alpha
        }
        while i < width * height:
            alpha = mapper[compression][0](data, ofs)
            ofs += mapper[compression][1]
            palette = _DDSPalette(data, ofs)  # generate palette
            chunk = struct.unpack_from("<I", data, ofs + 4)[0]  # read 4x4 pixel chunk
            for pos_y in range(y, y + 4):  # draw chunk
                for pos_x in range(x, x + 4):
                    color = palette.get(chunk & int('11', 2))  # 2 bits represent a palette index to look-up
                    if pos_x < width and pos_y < height:
                        set_pixel_callback(pos_x, pos_y, color.r, color.g, color.b, alpha.get(color))
                        i += 1
                    chunk = chunk >> 2  # shift color chunk by 1 pixels
            ofs += 8
            x += 4
            if x >= width:
                y += 4
                x = 0

    def __draw_uncompressed(self, set_pixel_callback: SetPixelCallback, data: bytes, width: int, height: int) -> None:
        """outputs the given uncompressed mip map via the given set_pixel callback."""
        ofs = x = y = 0
        bytes_per_pixel = 3 if self.__format == DDSFormat.UNCOMPRESSED24 else 4
        while ofs < width * height * bytes_per_pixel:
            color: int = struct.unpack_from("<I", data, ofs)[0]
            b: int = color & self.__color_mask.b
            g: int = (color & self.__color_mask.g) >> 8
            r: int = (color & self.__color_mask.r) >> 16
            a: int = (color & self.__color_mask.a) >> 24 if self.__format == DDSFormat.UNCOMPRESSED32 else 255
            set_pixel_callback(x, y, r, g, b, a)
            ofs += bytes_per_pixel
            x += 1
            if x >= width:
                y += 1
                x = 0

    def draw(self, set_pixel_callback: SetPixelCallback, mip: int) -> None:
        """
        outputs the given mip map via the given set_pixel callback.
        The callback must accept x and y coordinates and 4 8-bit RGBA values as parameters:
            set_pixel_callback(x: int, y: int, r: int, g: int, b: int, a: int) -> None
        """
        if 0 > mip >= len(self.__mips):
            raise IndexError
        data = self.__data[self.__mips[mip].offset:]
        width = self.__mips[mip].width
        height = self.__mips[mip].height
        if self.__format in [DDSFormat.UNCOMPRESSED32, DDSFormat.UNCOMPRESSED24]:
            self.__draw_uncompressed(set_pixel_callback, data, width, height)
        elif self.__format in [DDSFormat.DXT1, DDSFormat.DXGI_FORMAT_BC1_UNORM_SRGB]:
            self.__draw_compressed(set_pixel_callback, data, width, height, _DXTCompression.BC1)
        elif self.__format in [DDSFormat.DXT3, DDSFormat.DXGI_FORMAT_BC2_UNORM_SRGB]:
            self.__draw_compressed(set_pixel_callback, data, width, height, _DXTCompression.BC2)
        elif self.__format in [DDSFormat.DXT5, DDSFormat.DXGI_FORMAT_BC3_UNORM_SRGB]:
            self.__draw_compressed(set_pixel_callback, data, width, height, _DXTCompression.BC3)



__all__ = ['DDSImage', 'DDSException', 'DDSFormat']