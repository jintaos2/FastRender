#ifndef RENDER_H
#define RENDER_H

#include <stdio.h>
#include <algorithm>
#include "bitmap.h"
#include "shader.h"
#include "model.h"

class Render;

class FrameBuffer
{
public:
    uint32_t *fb_;
    int w_, h_;
    int size;
    float *z_buffer;
    FrameBuffer(int w, int h) : w_(w), h_(h), size(w * h)
    {
        fb_ = new uint32_t[size];
        z_buffer = new float[size];
    }
    ~FrameBuffer()
    {
        if (fb_)
        {
            delete[] fb_;
            fb_ = NULL;
        }
        if (z_buffer)
        {
            delete[] z_buffer;
            z_buffer = NULL;
        }
    }
    inline void fill(uint32_t color)
    {
        for (int i = 0; i < size; ++i)
        {
            fb_[i] = color;
            z_buffer[i] = FLT_MAX;
        }
    }
    inline void set_pixel(int x, int y, float z, uint32_t color)
    {
        int idx = y * w_ + x;
        // #pragma omp critical
        {
            if (z < z_buffer[idx])
            {
                z_buffer[idx] = z;
                fb_[idx] = color;
            }
        }
    }
    inline bool visiable(int x, int y, float z)
    {
        return z < z_buffer[y * w_ + x];
    }
};

struct Vertex2D
{
    float x;
    float y;
    float z;
    bool _out;
    int uv;
    int norm;
};
inline std::ostream &operator<<(std::ostream &os, const Vertex2D &a)
{
    os << "x:" << a.x << " y:" << a.y << " z:" << a.z << "  norm_ID:" << a.norm << "  uv_ID:" << a.uv;
    return os;
}
struct Obj
{
public:
    Model *model;
    Mat4x4f coordinate = matrix_set_identity();
    float scale = 1;
    Obj(Model *model_, Mat4x4f pose_, float scale_) : model(model_), coordinate(pose_), scale(scale_) {}
};

class RenderObj
{
public:
    FrameBuffer *frame_buffer_;
    Mat4x4f *camera;
    float *camera_scale;

    Model *model;
    Mat4x4f *obj_coordinate;
    float *obj_scale;

    Mat3x3f rotate_;
    Vec3f move_;
    std::vector<Vertex2D> vertex_;           // transformed
    std::vector<Vec3f> norms_;               // transformed
    std::vector<Vector<3, Vertex2D>> faces_; // clipped

    struct _Face
    {
        float coff11, coff12, coff21, coff22, x3, y3;
        float z1, z2, z3;
        Vec3f norm1, norm2, norm3;
        Vec2f uv1, uv2, uv3;
    };
    // |slope| < 1
    // input: (x0,y0) ,(x1,y1) and x1 <= x2, slope <= 1
    // output: step through x, output y
    struct Bresenham
    {
        bool UP;
        int dx;
        int dy;
        int D;
        int y_step = 1;
        int y, ret;
        bool flip = 1;
        Bresenham(int x0, int y0, int x1, int y1, bool UP) : UP(UP), dx(x1 - x0), dy(y1 - y0), y(y0), ret(y0)
        {
            if (dy < 0)
            {
                y_step = -1; // if k < 0, only change the y direction
                dy = -dy;    // dy = abs(dy)
            }
            if (dx >= dy)
                flip = 0;
            else
                std::swap(dx, dy); // flip
            D = -dx;               // error term
        }
        inline int step()
        {
            ret = y;
            if (flip)
            {
                while (1)
                {
                    y = y + y_step;
                    D = D + 2 * dy;
                    if (D > 0)
                    {
                        D = D - 2 * dx;
                        return UP ? ret : y - y_step;
                    }
                }
            }
            else
            {
                D = D + 2 * dy;
                if (D > 0)
                {
                    y = y + y_step;
                    D = D - 2 * dx;
                }
                return ret;
            }
        }
    };

    RenderObj(Render *render, Obj *obj);

    void render()
    {
        Mat4x4f transform = transformm_invert(*camera) * (*obj_coordinate); // 转换到相机空间.
        rotate_ = transformm_rotate(transform) * (*obj_scale);              // 提取旋转矩阵.
        move_ = transformm_move(transform);                                 // 提取位移.

        int mx = frame_buffer_->w_ / 2;
        int my = frame_buffer_->h_ / 2;
        float fx = frame_buffer_->w_ + 0.5;
        float fy = frame_buffer_->h_ + 0.5;

#pragma omp parallel for
        for (int i = 0; i < vertex_.size(); ++i)
        {
            Vec3f v = rotate_ * (model->_verts[i]) + move_;                 // 顶点坐标转换.
            float z = v.z / (*camera_scale);                                //
            Vertex2D vertex = {v.x / z + mx, v.y / z + my, z, false, 0, 0}; // 透视投影.
            if (vertex.z < 0.001 || vertex.x < -0.5 || vertex.y < -0.5 || vertex.x > fx || vertex.y > fy)
                vertex._out = true;
            vertex_[i] = vertex;
        }
#pragma omp parallel for
        for (int i = 0; i < norms_.size(); ++i)
        {
            norms_[i] = rotate_ * (model->_norms[i]); // 顶点法向量转换.
            // std::cout << "norm " << i << " after transfer\n"
            //           << norms_[i] << std::endl;
        }
        clip_faces();
        // #pragma omp parallel for
        for (int i = 0; i < faces_.size(); ++i)
        {
            Draw_triangle(faces_[i]);
        }
    }
    void clip_faces()
    {
        faces_.clear();

        // #pragma omp parallel for
        for (int i = 0; i < model->_faces.size(); ++i)
        {
            Vec3i p1 = model->_faces[i][0]; // 顶点索引 / 纹理坐标索引 / 顶点法向量索引.
            Vec3i p2 = model->_faces[i][1];
            Vec3i p3 = model->_faces[i][2];
            Vertex2D v1 = vertex_[p1.x];
            Vertex2D v2 = vertex_[p2.x];
            Vertex2D v3 = vertex_[p3.x];
            if (v1._out && v2._out && v3._out)
                continue;
            v1.uv = p1.y;
            v1.norm = p1.z;
            v2.uv = p2.y;
            v2.norm = p2.z;
            v3.uv = p3.y;
            v3.norm = p3.z;
            // sort p1, p2, p3
            if (v1.x > v2.x)
                std::swap(v1, v2);
            if (v1.x > v3.x)
                std::swap(v1, v3);
            if (v2.x > v3.x)
                std::swap(v2, v3);

            // #pragma omp critical
            {
                faces_.push_back({v1, v2, v3});
            }
        }
    }

    inline void Draw_triangle(Vector<3, Vertex2D> &face)
    {
        _Face f;
        // std::cout << "-------------------Draw_triangle----------------------: \n"
        //           << face.x << "\n"
        //           << face.y << "\n"
        //           << face.z << '\n';
        float x1f = face.x.x;
        float x2f = face.y.x;
        float x3f = face.z.x;
        float y1f = face.x.y;
        float y2f = face.y.y;
        float y3f = face.z.y;
        float z1 = face.x.z;
        float z2 = face.y.z;
        float z3 = face.z.z;
        int x1 = round(x1f);
        int x2 = round(x2f);
        int x3 = round(x3f);
        int y1 = round(y1f);
        int y2 = round(y2f);
        int y3 = round(y3f);
        // std::cout << "x1 x2 x3:" << x1 << "  " << x2 << "  " << x3 << std::endl;

        float coff3 = (y2f - y3f) * (x1f - x3f) + (x3f - x2f) * (y1f - y3f);
        f.coff11 = (y2f - y3f) / coff3;
        f.coff12 = (x3f - x2f) / coff3;
        f.coff21 = (y3f - y1f) / coff3;
        f.coff22 = (x1f - x3f) / coff3;
        f.x3 = x3f;
        f.y3 = y3f;
        f.z1 = z1;
        f.z2 = z2;
        f.z3 = z3;
        f.uv1 = model->_uv[face.x.uv];
        f.uv2 = model->_uv[face.y.uv];
        f.uv3 = model->_uv[face.z.uv];
        f.norm1 = model->_norms[face.x.norm];
        f.norm2 = model->_norms[face.y.norm];
        f.norm3 = model->_norms[face.z.norm];

        //int c = (y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1); // up, down, line
        if (x3 - x1 > 1)
        {
            float c = y2 - y1 - 1.0f * (y3 - y1) / (x3 - x1) * (x2 - x1);
            if (c > 0.9) // up
            {
                Bresenham l1(x1, y1, x3, y3, false);
                Bresenham l2(x1, y1, x2, y2, true);
                Bresenham l3(x2, y2, x3, y3, true);
                std::cout << "xxxyyy:  " << x1 << ' ' << x2 << ' ' << x3 << ' ' << y1 << ' ' << y2 << ' ' << y3 << " \n";
                int i = x1;
                for (; i < x2; ++i)
                {
                    // std::cout << "scanline: " << i << std::endl;
                    Draw_scanline(i, l2.step(), l1.step(), &f);
                }
                for (; i < x3; ++i)
                {
                    // std::cout << "scanline: " << i << std::endl;
                    Draw_scanline(i, l3.step(), l1.step(), &f);
                }
                if (x2 == x3)
                    Draw_scanline(x3, y2, y3, &f);
                else
                    Draw_scanline(x3, min(y3, l3.step()), max(y3, l1.step()), &f);
                return;
            }
            else if (c < -0.9) // down
            {
                Bresenham l1(x1, y1, x3, y3, true);
                Bresenham l2(x1, y1, x2, y2, false);
                Bresenham l3(x2, y2, x3, y3, false);
                // std::cout << "xxxyyy:  " << x1 << ' ' << x2 << ' ' << x3 << ' ' << y1 << ' ' << y2 << ' ' << y3 << " \n";
                int i = x1;
                for (; i < x2; ++i)
                {
                    // std::cout << "scanline: " << i << std::endl;
                    Draw_scanline(i, l1.step(), l2.step(), &f);
                }
                for (; i < x3; ++i)
                {
                    // std::cout << "scanline: " << i << std::endl;
                    Draw_scanline(i, l1.step(), l3.step(), &f);
                }
                if (x2 == x3)
                    Draw_scanline(x3, y3, y2, &f);
                else
                    Draw_scanline(x3, min(y3, l1.step()), max(y3, l3.step()), &f); // decide the end point
                return;
            }
        }
        Draw_line(x1, y1, x2, y2, &f);
        Draw_line(x1, y1, x3, y3, &f);
        Draw_line(x2, y2, x3, y3, &f);
    }
    inline void Draw_scanline(int x, int y1, int y2, _Face *f)
    {
        if (x < 0 || x >= frame_buffer_->w_)
            return;
        // if(y1 < y2) std::cout << "xxx" << std::endl;
        if (y1 > y2)
            std::swap(y1, y2);
        for (int y = y1; y <= y2; ++y)
        {
            // double frac1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) * 1.0 / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            // double frac2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) * 1.0 / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            // double frac3 = 1 - frac1 - frac2;
            if (y < 0 || y >= frame_buffer_->h_)
                continue;

            float frac1 = f->coff11 * (x - f->x3) + f->coff12 * (y - f->y3);
            float frac2 = f->coff21 * (x - f->x3) + f->coff22 * (y - f->y3);
            float frac3 = 1 - frac1 - frac2;
            frac1 /= f->z1;
            frac2 /= f->z2;
            frac3 /= f->z3;
            float z_ = 1.0f / (frac1 + frac2 + frac3);

            if (!frame_buffer_->visiable(x, y, z_))
                continue;

            Vec2f uv_pixel = (frac1 * f->uv1 + frac2 * f->uv2 + frac3 * f->uv3) * z_;
            uint32_t argb = model->diffuse(uv_pixel);
            argb = ((argb & 0x000000ff) << 16) | ((argb & 0x00ff0000) >> 16) | ((argb & 0xff00ff00));
            frame_buffer_->set_pixel(x, y, z_, argb);

            // std::cout << "fracs: " << frac1 << "  " << frac2 << "  " << frac3 << std::endl
            // std::cout << "uv_coord " << uv_pixel << " x:" << x << " y:" << y << std::endl;
            //printf("deepth z_ =%12.10f %12.10f %12.10f | %12.10f\n",p1->z, p2->z, p3->z,z_);
            // frame_buffer->SetPixel(x, y, z_, vector_to_color(model->diffuse(uv_pixel)));
        }
    }
    inline void Draw_line(int x0, int y0, int x1, int y1, _Face *f)
    {
        bool flip = false;
        if (std::abs(x0 - x1) < std::abs(y0 - y1))
        { // if dy > dx, swap x and y
            std::swap(x0, y0);
            std::swap(x1, y1);
            flip = true;
        }

        int dx = x1 - x0;
        int dy = y1 - y0;
        int D = -dx; // error
        int y_step = 1;
        if (dy < 0)
        {
            y_step = -1; // if k < 0, only change the y direction
            dy = -dy;    // dy = abs(dy)
        }
        int y = y0;
        for (int x = x0; x < x1 + 1; x++)
        {
            if (flip)
                Draw_scanline(y, x, x, f);
            else
                Draw_scanline(x, y, y, f);
            D = D + 2 * dy;
            if (D > 0)
            {
                y = y + y_step; // next y
                D = D - 2 * dx;
            }
        }
    }
};

class Render
{
public:
    FrameBuffer fb;
    Mat4x4f camera = matrix_set_identity();
    float camera_scale = 1;

    std::vector<RenderObj> obj_renders;

    Render(int w, int h) : fb(FrameBuffer(w, h)) {}
    void set_camera(Mat4x4f c, float scale)
    {
        camera = c;
        camera_scale = scale;
    }

    void move_camera_x(float dis)
    {
        camera.m[0][3] += camera.m[0][0] * dis;
        camera.m[1][3] += camera.m[1][0] * dis;
        camera.m[2][3] += camera.m[2][0] * dis;
    }
    void move_camera_y(float dis)
    {
        camera.m[0][3] += camera.m[0][1] * dis;
        camera.m[1][3] += camera.m[1][1] * dis;
        camera.m[2][3] += camera.m[2][1] * dis;
    }
    void move_camera_z(float dis)
    {
        camera.m[0][3] += camera.m[0][2] * dis;
        camera.m[1][3] += camera.m[1][2] * dis;
        camera.m[2][3] += camera.m[2][2] * dis;
    }
    void rotate_camera_left(float theta)
    {
        camera = camera * matrix_set_rotate(camera.m[0][1], camera.m[1][1], camera.m[2][1], theta);
    }
    void rotate_camera_up(float theta)
    {
        camera = camera * matrix_set_rotate(camera.m[0][0], camera.m[1][0], camera.m[2][0], theta);
    }
    void scale_camera(float scale)
    {
        float s = camera_scale * scale;
        float min_ = 0.000001;
        float max_ = 1000000;
        s = s < min_ ? min_ : s;
        camera_scale = s > max_ ? max_ : s;
    }
    void add_obj(Obj &obj)
    {
        obj_renders.push_back(RenderObj(this, &obj));
    }
    void render()
    {
        for (auto i : obj_renders)
        {
            i.render();
        }
    }
};

RenderObj::RenderObj(Render *render, Obj *obj)
{
    frame_buffer_ = &(render->fb);
    camera = &(render->camera);
    camera_scale = &(render->camera_scale);

    model = obj->model;
    obj_coordinate = &(obj->coordinate);
    obj_scale = &(obj->scale);

    vertex_.resize(model->_verts.size());
    norms_.resize(model->_norms.size());
    faces_.reserve(model->_faces.size());
}

#endif