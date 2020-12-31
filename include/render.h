#ifndef RENDER_H
#define RENDER_H

#include <stdio.h>
#include <algorithm>
#include <ctime>

#include "bitmap.h"
#include "model.h"

class Render;

class FrameBuffer
{
public:
    uint32_t *fb_;
    int w_, h_;
    std::vector<float *> z_buffers; // hierarchical z_buffer
    std::vector<int> z_buffer_w;    // width of each z_buffer
    std::vector<int> z_buffer_area;
    float *z_buffer0;
    int w0;
    int levels; // 3, 2, 1, ...

    FrameBuffer(int w, int h) : w_(w), h_(h)
    {
        if (w < h)
        {
            std::cout << "screen width < height \n";
            return;
        }
        fb_ = new uint32_t[w * h];

        int block_size = 1; // lowest level
        int level = 0;
        while (block_size < w)
        {
            block_size *= 2;
            level++;
        }
        z_buffers.resize(level);
        z_buffer_w.resize(level);
        z_buffer_area.resize(level);
        levels = level - 1;
        for (int i = levels; i >= 0; --i) // level = 3, size=8, 7*7 , first = 4, 2 1
        {
            block_size /= 2;
            int z_buffer_width = (w - 1) / block_size + 1;
            int z_buffer_height = (h - 1) / block_size + 1;
            z_buffer_w[i] = z_buffer_width;
            z_buffer_area[i] = z_buffer_width * z_buffer_height;
            z_buffers[i] = new float[z_buffer_width * z_buffer_height];
        }
        w0 = z_buffer_w[0];
        z_buffer0 = z_buffers[0];
    }
    ~FrameBuffer()
    {
        if (fb_)
        {
            delete[] fb_;
            fb_ = NULL;
        }
        for (int i = 0; i <= levels; ++i)
        {
            if (z_buffers[i])
                delete[] z_buffers[i];
        }
    }
    inline void fill(uint32_t color)
    {
        for (int i = levels; i >= 0; --i)
        {
            for (int j = 0; j < z_buffer_area[i]; ++j)
                z_buffers[i][j] = FLT_MAX;
        }
        for (int i = 0; i < w_ * h_; ++i)
            fb_[i] = color;
    }
    // 0 <= x1 < x2, 0 <= y1 < y2
    inline bool visiable_box(int x1, int y1, int x2, int y2, float z)
    {
        // find 4 blocks;
        // int i = 0;
        // for (; i <= levels; ++i)
        // {
        //     if ((x2 >> i) - (x1 >> i) < 2 && (y2 >> i) - (y1 >> i) < 2)
        //         break;
        // }
        // x1 >>= i;
        // x2 >>= i;
        // y1 >>= i;
        // y2 >>= i;
        // return z < z_buffers[i][z_buffer_sizes[i] * y1 + x1] ||
        //        z < z_buffers[i][z_buffer_sizes[i] * y2 + x1] ||
        //        z < z_buffers[i][z_buffer_sizes[i] * y1 + x2] ||
        //        z < z_buffers[i][z_buffer_sizes[i] * y2 + x2];
        // if (x1 > x2)
        //     std::swap(x1, x2);
        // if (y1 > y2)
        //     std::swap(x1, x2);
        int i = 0;
        while (x2 - x1 > 1 || y2 - y1 > 1)
        {
            x1 >>= 1;
            x2 >>= 1;
            y1 >>= 1;
            y2 >>= 1;
            i++;
        }
        return z < z_buffers[i][z_buffer_w[i] * y1 + x1] ||
               z < z_buffers[i][z_buffer_w[i] * y1 + x2] ||
               z < z_buffers[i][z_buffer_w[i] * y2 + x1] ||
               z < z_buffers[i][z_buffer_w[i] * y2 + x2];

        // for (int i = levels; i >= 0; --i) // 2, 1, 0
        // {
        //     int x = x1 >> i;
        //     int y = y1 >> i;
        //     if (x != (x2 >> i) || y != (y2 >> i))
        //     {
        //         return true;
        //     }
        //     // z_buffer farest <= nearest z, then not visiable
        //     else if (z_buffers[i][z_buffer_sizes[i] * y + x] <= z)
        //         return false;
        // }
        // return true;
    }
    //  y1 < y2
    inline bool visiable_scanline(int x, int y1, int y2, float z)
    {
        int i = 0;
        while (y2 - y1 > 1)
        {
            x >>= 1;
            y1 >>= 1;
            y2 >>= 1;
            i++;
        }
        return z < z_buffers[i][z_buffer_w[i] * y1 + x] ||
               z < z_buffers[i][z_buffer_w[i] * y2 + x];
    }
    inline bool visiable_pixel_hierarchical(int x, int y, float z)
    {
        return z < z_buffer0[w0 * y + x];
    }
    inline void set_pixel_hierarchical(int x, int y, float z, uint32_t color)
    {
        fb_[y * w_ + x] = color;
        z_buffer0[w0 * y + x] = z;
        float *zb_curr = z_buffer0;
        int w_curr = w0;
        for (int i = 1; i <= levels; ++i)
        {
            x &= (~1);
            y &= (~1);
            int idx = w_curr * y + x;
            float z00 = zb_curr[idx];
            float z10 = zb_curr[idx + 1];
            x >>= 1;
            y >>= 1;
            float z01 = zb_curr[idx + w_curr];
            float z11 = zb_curr[idx + w_curr + 1];
            if (z00 < z01)
                z00 = z01;
            if (z10 < z11)
                z10 = z11;
            if (z00 < z10)
                z00 = z10;
            zb_curr = z_buffers[i];
            w_curr = z_buffer_w[i];
            float &z_curr = zb_curr[w_curr * y + x];
            if (z00 < z_curr)
                z_curr = z00;
            else
                return;
        }
    }
    inline float get_z_hierarchical(int i, int x, int y)
    {
        return z_buffers[i][z_buffer_w[i] * y + x];
    }
    inline void set_pixel(int x, int y, float z, uint32_t color)
    {
        int idx = y * w0 + x;
        // #pragma omp critical
        {
            if (z < z_buffer0[idx])
            {
                z_buffer0[idx] = z;
                fb_[idx] = color;
            }
        }
    }
    inline bool visiable(int x, int y, float z)
    {
        return z < z_buffer0[y * w_ + x];
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
    float *pz1, *pz2, *pz3, *pz4; // hierarchical z_buffer pointers
    Face2D *f;
    bool operator<(const FaceID &a) const
    {
        return z1 < a.z1;
    }
};
struct Octree
{
    int x1, y1;
    float z1;
    int x2, y2;                                    // bounding box
    float z2;                                      // x1 < x2, y1 < y2 , z1  < z2
    Octree *n1, *n2, *n3, *n4, *n5, *n6, *n7, *n8; // childs;
    std::vector<Face2D *> faces;
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

    int X1, Y1, X2, Y2; //bounding box of the whole obj
    float Z1, Z2;

    Model *model;
    std::vector<int> z_buffer_w;
    std::vector<float *> z_buffers;
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
        X2 = 0;
        Y1 = h_ - 1;
        Y2 = 0;
        Z1 = FLT_MAX;
        Z2 = -1;
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
            float x1f = v31.x / z1 + mx;
            float x2f = v32.x / z2 + mx;
            float x3f = v33.x / z3 + mx;
            float y1f = v31.y / z1 + my;
            float y2f = v32.y / z2 + my;
            float y3f = v33.y / z3 + my;
            // 四舍五入.
            int x1 = x1f + 0.5;
            int x2 = x2f + 0.5;
            int x3 = x3f + 0.5;
            int y1 = y1f + 0.5;
            int y2 = y2f + 0.5;
            int y3 = y3f + 0.5;
            // bounding box: (x1, y1, z1) (x2, y2, z2)
            sort3(x1, x3, x2);
            sort3(y1, y3, y2);
            // 视锥剔除2.
            if ((x2 < 0) | (x1 >= w_) | (y2 < 0) | (y1 >= h_))
                continue;
            // 背面剔除.
            // if ((x2f - x1f) * (y3f - y2f) - (y2f - y1f) * (x3f - x2f) >= 0)
            //     continue;
            Face2D ff = {{x1f, y1f, z1, uvs[p1.y], p1.z},
                         {x2f, y2f, z2, uvs[p2.y], p2.z},
                         {x3f, y3f, z3, uvs[p3.y], p3.z},
                         p_diffusemap,
                         p_norms};
            sort3(ff.v1, ff.v2, ff.v3); // for bresenham
            faces_.push_back(ff);
            // push 之后.
            sort3(z1, z3, z2);
            // hierarchical z_buffer.
            x1 = between(0, w_ - 1, x1);
            x2 = between(0, w_ - 1, x2);
            y1 = between(0, h_ - 1, y1);
            y2 = between(0, h_ - 1, y2);
            // 更新obj bounding box;
            X1 = min(X1, x1);
            X2 = max(X2, x2);
            Y1 = min(Y1, y1);
            Y2 = max(Y2, y2);
            Z1 = min(Z1, z1);
            Z2 = max(Z2, z2);
            int level = 0;
            while (x2 - x1 > 1 || y2 - y1 > 1)
            {
                x1 >>= 1;
                x2 >>= 1;
                y1 >>= 1;
                y2 >>= 1;
                level++;
            }
            int s = z_buffer_w[level];
            float *pz = z_buffers[level];
            float *pz1 = pz + s * y1 + x1;
            float *pz2 = pz + s * y1 + x2;
            float *pz3 = pz + s * y2 + x1;
            float *pz4 = pz + s * y2 + x2;
            face_ids.push_back({z1, pz1, pz2, pz3, pz4, &faces_.back()});
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
    std::vector<RenderObj *> obj_renders;   // all objs
    // performance counter
    clock_t timer;
    int visiable_objs;
    int visiable_triangles;
    int visiable_scanlines;
    int visiable_pixels;

    Render(int w, int h) : fb(FrameBuffer(w, h)) {}
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
        int N = obj_renders.size();
        int n_faces = 0;

#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            obj_renders[i]->clip_faces();
        }
        for (auto i : obj_renders)
            n_faces += i->faces_.size();
        std::sort(std::begin(obj_renders), std::end(obj_renders),
                  [](RenderObj *a, RenderObj *b) -> bool { return a->Z1 < b->Z1; });

        std::cout << "time clip_faces = " << get_time_ms() << " ms\n";

        // #pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            RenderObj *c_ = obj_renders[i];
            if (c_->Z2 < 0 || !fb.visiable_box(c_->X1, c_->Y1, c_->X2, c_->Y2, c_->Z1))
                continue;
            for (auto j : c_->face_ids)
                Draw_triangle(j);
            visiable_objs++;
        }
        std::cout << "time Draw = " << get_time_ms() << " ms" << std::endl;
        std::cout << ">> faces:" << n_faces << "\t|obj:" << visiable_objs
                  << "\t|tiangle:" << visiable_triangles
                  << "\t|scanline:" << visiable_scanlines
                  << "\t|pixel:" << visiable_pixels << std::endl;
    }

    struct Face2D_Coeff
    {
        float ax, ay, ak, bx, by, bk, cx, cy, ck;
        float dx, dy;
    };
    inline void Draw_triangle(FaceID &face_id)
    {
        // 片元剔除.
        float min_z = face_id.z1;
        if ((min_z < *face_id.pz1 || min_z < *face_id.pz2 ||
             min_z < *face_id.pz3 || min_z < *face_id.pz4))
        {
            Face2D face = *face_id.f;
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
                    Draw_scanline(i, l1.step(), l2.step(), f, face);
                for (int i = x2; i < x3; ++i)
                    Draw_scanline(i, l1.step(), l3.step(), f, face);
                if (x2 == x3)
                    Draw_scanline(x3, y3, y2, f, face);
                else
                    Draw_scanline(x3, max(y3, l1.step()), min(y3, l3.step()), f, face);
            }
            else // down
            {
                Bresenham l1(x1, y1, x3, y3, true);
                Bresenham l2(x1, y1, x2, y2, false);
                Bresenham l3(x2, y2, x3, y3, false);
                int i = x1;
                for (; i < x2; ++i)
                    Draw_scanline(i, l2.step(), l1.step(), f, face);
                for (; i < x3; ++i)
                    Draw_scanline(i, l3.step(), l1.step(), f, face);
                if (x2 == x3)
                    Draw_scanline(x3, y2, y3, f, face);
                else
                    Draw_scanline(x3, max(y3, l3.step()), min(y3, l1.step()), f, face);
            }
        }
    }
    // y1 <= y2
    inline void Draw_scanline(int x, int y1, int y2, Face2D_Coeff &f, Face2D &face)
    {
        if (x < 0 || x >= fb.w_ || y2 < 0 || y1 >= fb.h_)
            return;
        visiable_scanlines += 1;
        y1 = between(0, fb.h_ - 1, y1);
        y2 = between(0, fb.h_ - 1, y2);

        float z1 = (x - face.v1.x) * f.dx + (y1 - face.v1.y) * f.dy + face.v1.z;
        // 扫描线剔除.
        // float z2 = z1 + (y2 - y1) * f.dy;
        // if ( (y2-y1 > 15) && !fb.visiable_scanline(x, y1, y2, min(z1, z2)))
        //     return;

        for (int y = y1; y <= y2; ++y)
        {
            float z_ = (y - y1) * f.dy + z1;
            // 像素剔除.
            if (!fb.visiable_pixel_hierarchical(x, y, z_))
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
            fb.set_pixel_hierarchical(x, y, z_, face.diffuse_map->Sample2D_easy(uv_x, uv_y));
        }
    }
    inline void Draw_pixel(int x, int y, Face2D_Coeff &f, Face2D &face)
    {
        if (x < 0 | x >= fb.w_ | y < 0 | y >= fb.h_)
            return;
        float z_ = (x - face.v1.x) * f.dx + (y - face.v1.y) * f.dy + face.v1.z;
        if (!fb.visiable_pixel_hierarchical(x, y, z_))
            return;
        visiable_pixels += 1;
        float frac1 = f.ax * x + f.ay * y + f.ak;
        float frac2 = f.bx * x + f.by * y + f.bk;
        float frac3 = f.cx * x + f.cy * y + f.ck;
        Vec2f &uv1 = face.v1.uv;
        Vec2f &uv2 = face.v2.uv;
        Vec2f &uv3 = face.v3.uv;
        float uv_x = (frac1 * uv1.x + frac2 * uv2.x + frac3 * uv3.x) * z_;
        float uv_y = (frac1 * uv1.y + frac2 * uv2.y + frac3 * uv3.y) * z_;
        fb.set_pixel_hierarchical(x, y, z_, face.diffuse_map->Sample2D_easy(uv_x, uv_y));
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
                Draw_pixel(y, x, f, face);
            else
                Draw_pixel(x, y, f, face);
            D = D + 2 * dy;
            if (D > 0)
            {
                y = y + y_step; // next y
                D = D - 2 * dx;
            }
        }
    }
};

RenderObj::RenderObj(Render *render, Obj *obj) : w_(render->fb.w_),
                                                 h_(render->fb.h_),
                                                 z_buffer_w(render->fb.z_buffer_w),
                                                 z_buffers(render->fb.z_buffers)
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
