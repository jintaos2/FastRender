#ifndef RENDER_H
#define RENDER_H

#include <stdio.h>
#include <omp.h>
#include <algorithm>
#include <ctime>

#include "bitmap.h"
#include "model.h"

class Render;

class FrameBuffer
{
public:
    uint32_t *fb_;
    float *z_buffer0;
    uint32_t *vaild;
    int w_, h_;
    FrameBuffer() {}
    FrameBuffer(int w, int h) : w_(w), h_(h)
    {
        fb_ = new uint32_t[w * h];
        z_buffer0 = new float[w * h];
        vaild = new uint32_t[w * h];
    }
    ~FrameBuffer()
    {
        if (fb_)
        {
            delete[] fb_;
            fb_ = NULL;
        }
        if (z_buffer0)
        {
            delete[] z_buffer0;
            z_buffer0 = NULL;
        }
        if (vaild)
        {
            delete[] vaild;
            vaild = NULL;
        }
    }
    inline void fill(uint32_t color)
    {
        for (int i = 0; i < w_ * h_; ++i)
        {
            fb_[i] = color;
            vaild[i] = 0; // z_buffer invaild
        }
    }
    inline void set_pixel(int x, int y, float z, uint32_t color)
    {
        int idx = y * w_ + x;
        vaild[idx] = 0xffffffff; // z_buffer vaild
        z_buffer0[idx] = z;
        fb_[idx] = color;
    }
    inline bool invisiable(int x, int y, float z)
    {
        int idx = y * w_ + x;
        return ((z >= z_buffer0[idx]) && vaild[idx]);
    }
};

struct Vertex2D
{
    float x;
    float y;
    float z;
    Vec2f uv;
    int norm;
    bool operator<(const Vertex2D &a) const
    {
        return x < a.x;
    }
    std::ostream &operator<<(std::ostream &os)
    {
        os << "x:" << x << " y:" << y << " z:" << z << "  norm_ID:" << norm;
        return os;
    }
};
struct Face2D
{
    Vertex2D v1, v2, v3;
    Bitmap *diffuse_map;
    std::vector<Vec3f> *norms;
};
struct FaceID
{
    float z1;
    Face2D *f;
    bool operator<(const FaceID &a) const
    {
        return z1 < a.z1;
    }
};

struct Obj
{
    Model *model;
    Mat4x4f coordinate;
    float scale = 1;
    Obj(Model *m_, Mat4x4f pose_, float s_) : model(m_), coordinate(pose_), scale(s_) {}
};

class RenderObj
{
public:
    int w_, h_; // size of screen
    Mat4x4f *camera;
    float *camera_scale;
    Mat4x4f *obj_coordinate;
    float *obj_scale;

    float X1, Y1, Z1;

    Model *model;
    std::vector<Vec3f> norms_;  // transformed
    std::vector<Face2D> faces_; // clipped faces
    std::vector<FaceID> face_ids;

    RenderObj(Render *render, Obj *obj);

    void clip_faces()
    {
        // initial state
        faces_.clear();
        face_ids.clear();
        X1 = w_ - 1;
        Y1 = h_ - 1;
        Z1 = FLT_MAX;
        Mat4x4f transform = transformm_invert(*camera) * (*obj_coordinate); // 转换到相机空间.
        Mat3x3f rotate_ = transformm_rotate(transform) * (*obj_scale);      // 提取旋转矩阵.
        Vec3f move_ = transformm_move(transform);                           // 提取位移.
        float scale_c = *camera_scale;                                      // 全局放大.
        int mx = w_ / 2;                                                    // 屏幕中心.
        int my = h_ / 2;
        // some ponters
        std::vector<Vec3f> *p_norms = &norms_;
        std::vector<Vec2f> &uvs = model->_uv;
        Bitmap *p_diffusemap = model->_diffusemap;

        // 顶点法向量转换.
        for (int i = 0; i < norms_.size(); ++i)
        {
            norms_[i] = rotate_ * (model->_norms[i]);
        }
        // 片元组装.
        for (int i = 0; i < model->_faces.size(); ++i)
        {
            Vector<3, Vec3i> &faceInt = model->_faces[i];
            // 顶点索引/纹理坐标索引/顶点法向量索引.
            Vec3i p1 = faceInt[0];
            Vec3i p2 = faceInt[1];
            Vec3i p3 = faceInt[2];
            // 顶点坐标转换.
            Vec3f v31 = rotate_ * (model->_verts[p1.x]) + move_;
            Vec3f v32 = rotate_ * (model->_verts[p2.x]) + move_;
            Vec3f v33 = rotate_ * (model->_verts[p3.x]) + move_;

            float z1 = v31.z / scale_c;
            float z2 = v32.z / scale_c;
            float z3 = v33.z / scale_c;
            // 视锥剔除1.
            if (z1 < 0.001 || z2 < 0.001 || z3 < 0.001)
                continue;
            // 透视投影.
            float x1 = v31.x / z1 + mx;
            float x2 = v32.x / z2 + mx;
            float x3 = v33.x / z3 + mx;
            float y1 = v31.y / z1 + my;
            float y2 = v32.y / z2 + my;
            float y3 = v33.y / z3 + my;
            // 背面剔除.
            if ((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2) >= 0)
                continue;
            Face2D ff = {{x1, y1, z1, uvs[p1.y], p1.z},
                         {x2, y2, z2, uvs[p2.y], p2.z},
                         {x3, y3, z3, uvs[p3.y], p3.z},
                         p_diffusemap,
                         p_norms};
            sort3(ff.v1, ff.v2, ff.v3); // for bresenham
            faces_.push_back(ff);
            // bounding box: (x1, y1, z1) (x2, y2, z2)
            sort3(x1, x3, x2);
            sort3(y1, y3, y2);
            // 视锥剔除2.
            if ((x2 < 0) | (x1 >= w_) | (y2 < 0) | (y1 >= h_))
                continue;
            sort3(z1, z3, z2);
            // 更新obj bounding box;
            X1 = min(X1, x1);
            Y1 = min(Y1, y1);
            Z1 = min(Z1, z1);
            face_ids.push_back({z1, &faces_.back()});
        }
        std::sort(face_ids.begin(), face_ids.end());
    }
};

class Render
{
public:
    Mat4x4f camera = matrix_set_identity(); // camera pose
    float camera_scale = 1;                 // scale of screen
    FrameBuffer fb;                         // frame buffer
    std::vector<FrameBuffer *> fbs;
    std::vector<RenderObj *> obj_renders; // all objs
    int n_threads;
    // performance counter
    clock_t timer;
    int visiable_objs;
    int visiable_triangles;
    int visiable_scanlines;
    int visiable_pixels;

    Render(int w, int h) : fb(FrameBuffer(w, h))
    {
        n_threads = omp_get_max_threads();
        for (int i = 0; i < n_threads; ++i)
        {
            fbs.push_back(new FrameBuffer(w, h));
        }
    }
    ~Render()
    {
        for (auto i : obj_renders)
            delete i;
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
        camera_scale = between(0.0001f, 10000.0f, s);
    }
    void add_obj(Obj *obj)
    {
        obj_renders.push_back(new RenderObj(this, obj));
    }
    double get_time_ms()
    {
        double ret = (double)(clock() - timer) * 1000.0 / CLOCKS_PER_SEC;
        timer = clock();
        return ret;
    }
    void render()
    {
        std::cout << "=================================== new frame =====\n";

        timer = clock();
        visiable_objs = 0;
        visiable_triangles = 0;
        visiable_scanlines = 0;
        visiable_pixels = 0;
        int n_faces = 0;

        int N = obj_renders.size();
        int n_obj = (N - 1) / n_threads + 1;

        // omp_set_num_threads(n_threads);
#pragma omp parallel for num_threads(6)
        for (int i = 0; i < N; ++i)
        {
            obj_renders[i]->clip_faces();
        }

        for (auto i : obj_renders)
            n_faces += i->faces_.size();
        std::sort(std::begin(obj_renders), std::end(obj_renders),
                  [](RenderObj *a, RenderObj *b) -> bool { return a->X1 < b->X1; });
        std::cout << "time clip_faces = " << get_time_ms() << " ms\n";

        omp_set_num_threads(n_threads);
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int n_start = thread_id * n_obj;
            int n_end = min(N, n_start + n_obj);
            for (int i = n_start; i < n_end; ++i)
            {
                for (auto f : obj_renders[i]->face_ids)
                    Draw_triangle(f, fbs[thread_id]);
            }
        }
        std::cout << "time Draw = " << get_time_ms() << " ms" << std::endl;
        std::cout << "faces:" << n_faces << "\tdraw->obj:" << visiable_objs
                  << "\triangle:" << visiable_triangles
                  << "\tscanline:" << visiable_scanlines
                  << "\tpixel:" << visiable_pixels << std::endl;
    }

    struct Face2D_Coeff
    {
        float ax, ay, ak, bx, by, bk, cx, cy, ck;
        float dx, dy;
    };
    inline void Draw_triangle(FaceID f_, FrameBuffer *fb_)
    {
        Face2D &face = *f_.f;
        float x1f = face.v1.x;
        float x2f = face.v2.x;
        float x3f = face.v3.x;
        float y1f = face.v1.y;
        float y2f = face.v2.y;
        float y3f = face.v3.y;
        float z2 = face.v2.z;
        float z3 = face.v3.z;
        int x1 = x1f + 0.5;
        int x2 = x2f + 0.5;
        int x3 = x3f + 0.5;
        int y1 = y1f + 0.5;
        int y2 = y2f + 0.5;
        int y3 = y3f + 0.5;

        int c = (y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1); // up, down, line
        visiable_triangles += 1;

        if (c == 0)
            return;
        float coeff1 = (y2f - y3f) * (x1f - x3f) + (x3f - x2f) * (y1f - y3f);
        float dz23 = (z2 - z3) / coeff1;
        float dz12 = (face.v1.z - z2) / coeff1;
        float cz11 = coeff1 * face.v1.z;
        float cz12 = coeff1 * z2;
        float cz13 = coeff1 * z3;
        Face2D_Coeff f = {(y2f - y3f) / cz11,
                          (x3f - x2f) / cz11,
                          (x2f * y3f - x3f * y2f) / cz11,
                          (y3f - y1f) / cz12,
                          (x1f - x3f) / cz12,
                          (x3f * y1f - x1f * y3f) / cz12,
                          (y1f - y2f) / cz13,
                          (x2f - x1f) / cz13,
                          (x1f * y2f - x2f * y1f) / cz13,
                          (y2f - y1f) * dz23 + (y2f - y3f) * dz12,
                          (x1f - x2f) * dz23 + (x3f - x2f) * dz12};
        if (c < 0) // up
        {
            Bresenham l1(x1, y1, x3, y3, false);
            Bresenham l2(x1, y1, x2, y2, true);
            Bresenham l3(x2, y2, x3, y3, true);
            for (int i = x1; i < x2; ++i)
                Draw_scanline(i, l1.step(), l2.step(), f, face, fb_);
            for (int i = x2; i < x3; ++i)
                Draw_scanline(i, l1.step(), l3.step(), f, face, fb_);
            if (x2 == x3)
                Draw_scanline(x3, y3, y2, f, face, fb_);
            else
                Draw_scanline(x3, max(y3, l1.step()), min(y3, l3.step()), f, face, fb_);
        }
        else // down
        {
            Bresenham l1(x1, y1, x3, y3, true);
            Bresenham l2(x1, y1, x2, y2, false);
            Bresenham l3(x2, y2, x3, y3, false);
            int i = x1;
            for (; i < x2; ++i)
                Draw_scanline(i, l2.step(), l1.step(), f, face, fb_);
            for (; i < x3; ++i)
                Draw_scanline(i, l3.step(), l1.step(), f, face, fb_);
            if (x2 == x3)
                Draw_scanline(x3, y2, y3, f, face, fb_);
            else
                Draw_scanline(x3, max(y3, l3.step()), min(y3, l1.step()), f, face, fb_);
        }
    }
    // y1 <= y2
    inline void Draw_scanline(int x, int y1, int y2, Face2D_Coeff &f, Face2D &face, FrameBuffer *fb_)
    {
        if (x < 0 || x >= fb.w_ || y2 < 0 || y1 >= fb.h_)
            return;
        visiable_scanlines += 1;
        y1 = between(0, fb.h_ - 1, y1);
        y2 = between(0, fb.h_ - 1, y2);

        float z1 = (x - face.v1.x) * f.dx + (y1 - face.v1.y) * f.dy + face.v1.z;

        for (int y = y1; y <= y2; ++y)
        {
            float z_ = (y - y1) * f.dy + z1;
            // 像素剔除.
            if (fb_->invisiable(x, y, z_))
                continue;
            visiable_pixels += 1;
            float frac1 = f.ax * x + f.ay * y + f.ak;
            float frac2 = f.bx * x + f.by * y + f.bk;
            float frac3 = f.cx * x + f.cy * y + f.ck;
            Vec2f &uv1 = face.v1.uv;
            Vec2f &uv2 = face.v2.uv;
            Vec2f &uv3 = face.v3.uv;
            float uv_x = (frac1 * uv1.x + frac2 * uv2.x + frac3 * uv3.x) * z_;
            float uv_y = (frac1 * uv1.y + frac2 * uv2.y + frac3 * uv3.y) * z_;
            fb_->set_pixel(x, y, z_, face.diffuse_map->Sample2D_easy(uv_x, uv_y));
        }
    }

    struct Bresenham
    {
        uint32_t ret_;
        int dx;
        int dy;
        int D;
        int y_step;
        int y, ret;
        bool flip = true;
        Bresenham(int x0, int y0, int x1, int y1, bool UP) : dx(x1 - x0), dy(y1 - y0), y(y0), ret(y0)
        {
            int mask = (dy > -1);
            y_step = (mask << 1) - 1; // if k < 0, only change the y direction

            mask = dy >> 31;
            dy = (dy + mask) ^ mask; // dy = abs(dy)

            flip = dx < dy;
            if (flip)
                std::swap(dx, dy); // flip
            D = -dx;               // error term
            dy *= 2;
            dx *= 2;
            // if (up && y_step = 1 || down && y_step = -1), y-y_step
            // ret_ = (UP ^ (y_step > 0)) ? 0xffffffff : 0;
            mask = UP ^ (y_step > 0);
            ret_ = -mask;
        }
        inline int step()
        {
            ret = y;
            while (flip)
            {
                y = y + y_step;
                D = D + dy;
                if (D > 0)
                {
                    D = D - dx;
                    return ret_ & ret | (~ret_) & (y - y_step);
                }
            }
            D = D + dy;
            int mask = D > 0;
            mask = -mask;
            y = y + (mask & y_step);
            D = D - (mask & dx);
            return ret;
        }
    };
};

RenderObj::RenderObj(Render *render, Obj *obj) : w_(render->fb.w_),
                                                 h_(render->fb.h_)
{
    camera = &(render->camera);
    camera_scale = &(render->camera_scale);
    model = obj->model;
    obj_coordinate = &(obj->coordinate);
    obj_scale = &(obj->scale);

    norms_.resize(model->_norms.size());
    faces_.reserve(model->_faces.size());
    face_ids.reserve(model->_faces.size());
}

#endif
