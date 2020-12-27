#ifndef RENDER_H
#define RENDER_H

#include <stdio.h>
#include <algorithm>
#include <ctime>

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
struct Face2D
{
    Vertex2D v1, v2, v3;
    std::vector<Vec3f> *norms;
    std::vector<Vec2f> *uvs;
    Bitmap *diffuse_map;
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
    std::vector<Vertex2D> vertex_; // transformed
    std::vector<Vec3f> norms_;     // transformed
    std::vector<Face2D> *faces_;   // clipped

    RenderObj(Render *render, Obj *obj);

    void transform()
    {
        Mat4x4f transform = transformm_invert(*camera) * (*obj_coordinate); // 转换到相机空间.
        rotate_ = transformm_rotate(transform) * (*obj_scale);              // 提取旋转矩阵.
        move_ = transformm_move(transform);                                 // 提取位移.

        int mx = frame_buffer_->w_ / 2;
        int my = frame_buffer_->h_ / 2;
        float fx = frame_buffer_->w_ + 0.5;
        float fy = frame_buffer_->h_ + 0.5;

        // #pragma omp parallel for
        for (int i = 0; i < vertex_.size(); ++i)
        {
            Vec3f v = rotate_ * (model->_verts[i]) + move_;                 // 顶点坐标转换.
            float z = v.z / (*camera_scale);                                //
            Vertex2D vertex = {v.x / z + mx, v.y / z + my, z, false, 0, 0}; // 透视投影.
            vertex._out = vertex.z < 0.001 || vertex.x < -0.5 ||
                          vertex.y < -0.5 || vertex.x > fx || vertex.y > fy;
            vertex_[i] = vertex;
        }
        for (int i = 0; i < norms_.size(); ++i)
            norms_[i] = rotate_ * (model->_norms[i]); // 顶点法向量转换.
    }
    void clip_faces()
    {
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
                continue; // v2.y - v1.y      v2.x - v1.x
            float ax = v2.x - v1.x;
            float ay = v2.y - v1.y;
            float bx = v3.x - v2.x;
            float by = v3.y - v2.y;
            // if ((v2.x - v1.x) * (v3.y - v2.y) - (v2.y - v1.y) * (v3.x - v2.x) > 0)
            //     continue; // 背面剔除.
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
                faces_->push_back({v1, v2, v3, &norms_, &model->_uv, model->_diffusemap});
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

    std::vector<RenderObj *> obj_renders;
    std::vector<Face2D> faces_;

    clock_t timer;

    Render(int w, int h) : fb(FrameBuffer(w, h)) {}
    ~Render()
    {
        for (auto i : obj_renders)
        {
            if (i)
                delete i;
        }
        obj_renders.clear();
    }

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
    void add_obj(Obj *obj)
    {
        obj_renders.push_back(new RenderObj(this, obj));
    }

    struct Bresenham
    {
        uint32_t ret_;
        int dx;
        int dy;
        int D;
        int y_step = 1;
        int y, ret;
        bool flip = true;
        Bresenham(int x0, int y0, int x1, int y1, bool UP) : dx(x1 - x0), dy(y1 - y0), y(y0), ret(y0)
        {
            if (dy < 0)
            {
                y_step = -1; // if k < 0, only change the y direction
                dy = -dy;    // dy = abs(dy)
            }
            if (dx >= dy)
                flip = false;
            else
                std::swap(dx, dy); // flip
            D = -dx;               // error term
                                   // if (up && y_step = 1 || down && y_step = -1), y-y_step
            ret_ = UP ^ (y_step > 0) ? 0xffffffff : 0;
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
                        return ret_ & ret | (~ret_) & (y - y_step);
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

    void render()
    {
        timer = clock();
        for (auto i : obj_renders)
            i->transform();
        faces_.clear();
        for (auto i : obj_renders)
            i->clip_faces();
        std::cout << "time clip_faces = " << (double)(clock() - timer) * 1000.0 / CLOCKS_PER_SEC << " ms\tn_faces: " << faces_.size() << std::endl;
        timer = clock();
        // #pragma omp parallel for
        for (int i = 0; i < faces_.size(); ++i)
        {
            Draw_triangle(faces_[i]);
        }
        std::cout << "time Draw_triangle = " << (double)(clock() - timer) * 1000.0 / CLOCKS_PER_SEC << " ms" << std::endl;
    }

    struct Face2D_Coeff
    {
        float ax, ay, ak, bx, by, bk, cx, cy, ck;
    };
    inline void Draw_triangle(Face2D &face)
    {
        float x1f = face.v1.x;
        float x2f = face.v2.x;
        float x3f = face.v3.x;
        float y1f = face.v1.y;
        float y2f = face.v2.y;
        float y3f = face.v3.y;
        float z1 = face.v1.z;
        float z2 = face.v2.z;
        float z3 = face.v3.z;
        int x1 = x1f + 0.5;
        int x2 = x2f + 0.5;
        int x3 = x3f + 0.5;
        int y1 = y1f + 0.5;
        int y2 = y2f + 0.5;
        int y3 = y3f + 0.5;

        // std::cout << "x1 x2 x3:" << x1 << "  " << x2 << "  " << x3 << std::endl;

        float coeff1 = (y2f - y3f) * (x1f - x3f) + (x3f - x2f) * (y1f - y3f);
        Face2D_Coeff f = {(y2f - y3f) / coeff1 / z1,
                          (x3f - x2f) / coeff1 / z1,
                          (x2f * y3f - x3f * y2f) / coeff1 / z1,
                          (y3f - y1f) / coeff1 / z2,
                          (x1f - x3f) / coeff1 / z2,
                          (x3f * y1f - x1f * y3f) / coeff1 / z2,
                          (y1f - y2f) / coeff1 / z3,
                          (x2f - x1f) / coeff1 / z3,
                          (x1f * y2f - x2f * y1f) / coeff1 / z3};

        int c = (y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1); // up, down, line
        if (c < 0)                                             // up
        {
            Bresenham l1(x1, y1, x3, y3, false);
            Bresenham l2(x1, y1, x2, y2, true);
            Bresenham l3(x2, y2, x3, y3, true);
            // std::cout << "xxxyyy:  " << x1 << ' ' << x2 << ' ' << x3 << ' ' << y1 << ' ' << y2 << ' ' << y3 << " \n";
            for (int i = x1; i < x2; ++i)
                Draw_scanline(i, l2.step(), l1.step(), f, face);
            for (int i = x2; i < x3; ++i)
                Draw_scanline(i, l3.step(), l1.step(), f, face);
            if (x2 == x3)
                Draw_scanline(x3, y2, y3, f, face);
            else
                Draw_scanline(x3, min(y3, l3.step()), max(y3, l1.step()), f, face);
        }
        else // down
        {
            Bresenham l1(x1, y1, x3, y3, true);
            Bresenham l2(x1, y1, x2, y2, false);
            Bresenham l3(x2, y2, x3, y3, false);
            int i = x1;
            for (; i < x2; ++i)
                Draw_scanline(i, l1.step(), l2.step(), f, face);
            for (; i < x3; ++i)
                Draw_scanline(i, l1.step(), l3.step(), f, face);
            if (x2 == x3)
                Draw_scanline(x3, y3, y2, f, face);
            else
                Draw_scanline(x3, min(y3, l1.step()), max(y3, l3.step()), f, face); // decide the end point
        }
    }
    // y1 >= y2
    inline void Draw_scanline(int x, int y1, int y2, Face2D_Coeff &f, Face2D &face)
    {
        if (x < 0 || x >= fb.w_ || y1 < y2)
            return;
        for (int y = y2; y <= y1; ++y)
        {
            if (y < 0 || y >= fb.h_)
                continue;

            float frac1 = f.ax * x + f.ay * y + f.ak;
            float frac2 = f.bx * x + f.by * y + f.bk;
            float frac3 = f.cx * x + f.cy * y + f.ck;
            float z_ = 1.0f / (frac1 + frac2 + frac3);
            // if (!fb.visiable(x, y, z_))
            //     continue;

            uint32_t argb = 0;
            Vec2f uv1 = face.uvs->at(face.v1.uv);
            Vec2f uv2 = face.uvs->at(face.v2.uv);
            Vec2f uv3 = face.uvs->at(face.v3.uv);
            Vec2f uv_pixel = (frac1 * uv1 + frac2 * uv2 + frac3 * uv3) * z_;
            argb = face.diffuse_map->Sample2D(uv_pixel);
            argb = ((argb & 0x000000ff) << 16) | ((argb & 0x00ff0000) >> 16) | ((argb & 0xff00ff00));
            fb.set_pixel(x, y, z_, argb);
            // fb.set_pixel(2,2, z_,uv_pixel.x );

            // std::cout << "fracs: " << frac1 << "  " << frac2 << "  " << frac3 << std::endl
            // std::cout << "uv_coord " << uv_pixel << " x:" << x << " y:" << y << std::endl;
        }
    }
    inline void Draw_line(int x0, int y0, int x1, int y1, Face2D_Coeff &f, Face2D &face)
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
                Draw_scanline(y, x, x, f, face);
            else
                Draw_scanline(x, y, y, f, face);
            D = D + 2 * dy;
            if (D > 0)
            {
                y = y + y_step; // next y
                D = D - 2 * dx;
            }
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
    faces_ = &render->faces_;

    vertex_.resize(model->_verts.size());
    norms_.resize(model->_norms.size());
    faces_->reserve(faces_->size() + model->_faces.size());
}

#endif
