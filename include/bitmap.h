#ifndef BITMAP_H
#define BITMAP_H

#include "matrix.h"

//---------------------------------------------------------------------
// 位图库：用于加载/保存图片，画点，画线，颜色读取
//---------------------------------------------------------------------
struct _Color
{
    union
    {
        struct
        {
            uint8_t r, g, b, a;
        };
        struct
        {
            uint8_t B, G, R, A;
        };
        uint32_t argb = 0x000000ff;
    };
};
class Bitmap
{
public:
    inline virtual ~Bitmap()
    {
        if (_bits)
            delete[] _bits;
        _bits = NULL;
    }
    inline Bitmap(int width, int height) : _w(width), _h(height)
    {
        _pitch = width * 4;
        _bits = new uint8_t[_pitch * _h];
    }

    inline Bitmap(const Bitmap &src) : _w(src._w), _h(src._h), _pitch(src._pitch)
    {
        _bits = new uint8_t[_pitch * _h];
        memcpy(_bits, src._bits, _pitch * _h);
    }

    inline Bitmap(const char *filename)
    {
        Bitmap *tmp = LoadFile(filename);
        if (tmp == NULL)
        {
            std::string msg = "load failed: ";
            msg.append(filename);
            throw std::runtime_error(msg);
        }
        _w = tmp->_w;
        _h = tmp->_h;
        _pitch = tmp->_pitch;
        _bits = tmp->_bits;
        
        tmp->_bits = NULL;
        delete tmp;
    }

public:
    inline int GetW() const { return _w; }
    inline int GetH() const { return _h; }
    inline int GetPitch() const { return _pitch; }
    inline uint8_t *GetBits() { return _bits; }
    inline const uint8_t *GetBits() const { return _bits; }
    inline uint8_t *GetLine(int y) { return _bits + _pitch * y; }
    inline const uint8_t *GetLine(int y) const { return _bits + _pitch * y; }

public:
    inline uint32_t GetPixel(int x, int y) const
    {
        uint32_t color = 0;
        if (x >= 0 && x < _w && y >= 0 && y < _h)
        {
            memcpy(&color, _bits + y * _pitch + x * 4, sizeof(uint32_t));
        }
        return color;
    }
    inline void SetPixel(int x, int y, uint32_t color)
    {
        if (x >= 0 && x < _w && y >= 0 && y < _h)
        {
            memcpy(_bits + y * _pitch + x * 4, &color, sizeof(uint32_t));
        }
    }
    struct BITMAPINFOHEADER
    { // bmih
        uint32_t biSize;
        uint32_t biWidth;
        int32_t biHeight;
        uint16_t biPlanes;
        uint16_t biBitCount;
        uint32_t biCompression;
        uint32_t biSizeImage;
        uint32_t biXPelsPerMeter;
        uint32_t biYPelsPerMeter;
        uint32_t biClrUsed;
        uint32_t biClrImportant;
    };

    // 读取 BMP 图片，支持 24/32 位两种格式.
    inline static Bitmap *LoadFile(const char *filename)
    {
        FILE *fp = fopen(filename, "rb");
        if (fp == NULL)
            return NULL;
        BITMAPINFOHEADER info;
        uint8_t header[14];
        int hr = (int)fread(header, 1, 14, fp);
        if (hr != 14)
        {
            fclose(fp);
            return NULL;
        }
        if (header[0] != 0x42 || header[1] != 0x4d)
        {
            fclose(fp);
            return NULL;
        }
        hr = (int)fread(&info, 1, sizeof(info), fp);
        if (hr != 40)
        {
            fclose(fp);
            return NULL;
        }
        if (info.biBitCount != 24 && info.biBitCount != 32)
        {
            fclose(fp);
            return NULL;
        }
        Bitmap *bmp = new Bitmap(info.biWidth, info.biHeight);
        uint32_t offset;
        memcpy(&offset, header + 10, sizeof(uint32_t));
        fseek(fp, offset, SEEK_SET);
        uint32_t pixelsize = (info.biBitCount + 7) / 8;
        uint32_t pitch = (pixelsize * info.biWidth + 3) & (~3);
        for (int y = 0; y < (int)info.biHeight; y++)
        {
            uint8_t *line = bmp->GetLine(info.biHeight - 1 - y);
            for (int x = 0; x < (int)info.biWidth; x++, line += 4)
            {
                line[3] = 255;
                fread(line, pixelsize, 1, fp);
            }
            fseek(fp, pitch - info.biWidth * pixelsize, SEEK_CUR);
        }
        fclose(fp);
        return bmp;
    }

    // 保存 BMP 图片.
    inline bool SaveFile(const char *filename, bool withAlpha = false) const
    {
        FILE *fp = fopen(filename, "wb");
        if (fp == NULL)
            return false;
        BITMAPINFOHEADER info;
        uint32_t pixelsize = (withAlpha) ? 4 : 3;
        uint32_t pitch = (GetW() * pixelsize + 3) & (~3);
        info.biSizeImage = pitch * GetH();
        uint32_t bfSize = 54 + info.biSizeImage;
        uint32_t zero = 0, offset = 54;
        fputc(0x42, fp);
        fputc(0x4d, fp);
        fwrite(&bfSize, 4, 1, fp);
        fwrite(&zero, 4, 1, fp);
        fwrite(&offset, 4, 1, fp);
        info.biSize = 40;
        info.biWidth = GetW();
        info.biHeight = GetH();
        info.biPlanes = 1;
        info.biBitCount = (withAlpha) ? 32 : 24;
        info.biCompression = 0;
        info.biXPelsPerMeter = 0xb12;
        info.biYPelsPerMeter = 0xb12;
        info.biClrUsed = 0;
        info.biClrImportant = 0;
        fwrite(&info, sizeof(info), 1, fp);
        // printf("pitch=%d %d\n", (int)pitch, info.biSizeImage);
        for (int y = 0; y < GetH(); y++)
        {
            const uint8_t *line = GetLine(info.biHeight - 1 - y);
            uint32_t padding = pitch - GetW() * pixelsize;
            for (int x = 0; x < GetW(); x++, line += 4)
            {
                fwrite(line, pixelsize, 1, fp);
            }
            for (int i = 0; i < (int)padding; i++)
                fputc(0, fp);
        }
        fclose(fp);
        return true;
    }

    // 双线性插值.
    inline uint32_t SampleBilinear(float x, float y) const
    {
        int32_t fx = (int32_t)(x * 0x10000);
        int32_t fy = (int32_t)(y * 0x10000);
        int32_t x1 = Between(0, _w - 1, fx >> 16);
        int32_t y1 = Between(0, _h - 1, fy >> 16);
        int32_t x2 = Between(0, _w - 1, x1 + 1);
        int32_t y2 = Between(0, _h - 1, y1 + 1);
        int32_t dx = (fx >> 8) & 0xff;
        int32_t dy = (fy >> 8) & 0xff;
        if (_w <= 0 || _h <= 0)
            return 0xffff0000;
        uint32_t c00 = GetPixel(x1, y1);
        uint32_t c01 = GetPixel(x2, y1);
        uint32_t c10 = GetPixel(x1, y2);
        uint32_t c11 = GetPixel(x2, y2);
        return BilinearInterp(c00, c01, c10, c11, dx, dy);
    }

    // 纹理采样.
    inline uint32_t Sample2D(float u, float v)
    {
        float x = u * _w + 0.5f;
        float y = v * _h + 0.5f;
        int x1 = x;
        int y1 = y;
        float frac1 = (x - x1);
        float frac2 = (y - y1);
        float f00 = frac1 * frac2;
        float f01 = frac1 * (1 - frac2);
        float f10 = (1 - frac1) * frac2;
        float f11 = (1 - frac1) * (1 - frac2);
        x1 = x1 < 0 ? 0 : x1;
        x1 = x1 > _w - 2 ? _w - 2 : x1;
        y1 = y1 < 0 ? 0 : y1;
        y1 = y1 > _h - 2 ? _h - 2 : y1;
        return frac1 + frac2 + f00 + f01 + f10 + f11 + x1 + y1;
        uint32_t c00 = get_color(x1, y1);
        uint32_t c10 = get_color(x1 + 1, y1);
        uint32_t c01 = get_color(x1, y1 + 1);
        uint32_t c11 = get_color(x1 + 1, y1 + 1);
        uint32_t B = (c00 & 0x000000ff) * f00 + (c01 & 0x000000ff) * f01 + (c10 & 0x000000ff) * f10 + (c11 & 0x000000ff) * f11;
        uint32_t G = (c00 & 0x0000ff00) * f00 + (c01 & 0x0000ff00) * f01 + (c10 & 0x0000ff00) * f10 + (c11 & 0x0000ff00) * f11;
        uint32_t R = (c00 & 0x00ff0000) * f00 + (c01 & 0x00ff0000) * f01 + (c10 & 0x00ff0000) * f10 + (c11 & 0x00ff0000) * f11;
        return c11;
        return (B & 0x000000ff) << 16 | (G & 0x0000ff00) | (R & 0x00ff0000) >> 16 | 0xff000000;
    }
    inline uint32_t get_color(int x, int y)
    {
        return ((uint32_t *)_bits)[y * _w + x];
    }
    inline uint32_t Sample2D_easy(float u, float v)
    {
        uint32_t x1 = u * _w;
        uint32_t y1 = v * _h;
        x1 = x1 > _w - 1 ? _w - 1 : x1;
        y1 = y1 > _h - 1 ? _h - 1 : y1;
        uint32_t bgra = ((uint32_t*)_bits)[y1 * _w + x1];
        return (bgra & 0x000000ff) << 16 | (bgra >> 16) & 0x000000ff | bgra & 0xff00ff00;
    }
    // 纹理采样：直接传入 Vec2f
    inline uint32_t Sample2D(const Vec2f &uv)
    {
        return Sample2D(uv.x, uv.y);
    }

    // 上下反转.
    inline void FlipVertical()
    {
        uint8_t *buffer = new uint8_t[_pitch];
        for (int i = 0, j = _h - 1; i < j; i++, j--)
        {
            memcpy(buffer, GetLine(i), _pitch);
            memcpy(GetLine(i), GetLine(j), _pitch);
            memcpy(GetLine(j), buffer, _pitch);
        }
        delete[] buffer;
    }

protected:
    // 双线性插值计算：给出四个点的颜色，以及坐标偏移，计算结果.
    inline static uint32_t BilinearInterp(uint32_t tl, uint32_t tr,
                                          uint32_t bl, uint32_t br, int32_t distx, int32_t disty)
    {
        uint32_t f, r;
        int32_t distxy = distx * disty;
        int32_t distxiy = (distx << 8) - distxy; /* distx * (256 - disty) */
        int32_t distixy = (disty << 8) - distxy; /* disty * (256 - distx) */
        int32_t distixiy = 256 * 256 - (disty << 8) - (distx << 8) + distxy;
        r = (tl & 0x000000ff) * distixiy + (tr & 0x000000ff) * distxiy + (bl & 0x000000ff) * distixy + (br & 0x000000ff) * distxy;
        f = (tl & 0x0000ff00) * distixiy + (tr & 0x0000ff00) * distxiy + (bl & 0x0000ff00) * distixy + (br & 0x0000ff00) * distxy;
        r |= f & 0xff000000;
        tl >>= 16;
        tr >>= 16;
        bl >>= 16;
        br >>= 16;
        r >>= 16;
        f = (tl & 0x000000ff) * distixiy + (tr & 0x000000ff) * distxiy + (bl & 0x000000ff) * distixy + (br & 0x000000ff) * distxy;
        r |= f & 0x00ff0000;
        f = (tl & 0x0000ff00) * distixiy + (tr & 0x0000ff00) * distxiy + (bl & 0x0000ff00) * distixy + (br & 0x0000ff00) * distxy;
        r |= f & 0xff000000;
        return r;
    }

public:
    int32_t _w;
    int32_t _h;
    int32_t _pitch;
    uint8_t *_bits;
    uint32_t *color_;
};

#endif
