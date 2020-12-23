#ifndef RENDER_H
#define RENDER_H

#include <stdio.h>
#include "bitmap.h"
#include "shader.h"
#include "model.h"

class Render;

inline static Vec3d interp(Vec3f v1, Vec3f v2, double frac)
{
    Vec3d ret;
    ret.x = v1.x * frac + (1 - frac) * v2.x;
    ret.y = v1.y * frac + (1 - frac) * v2.y;
    ret.z = v1.z * frac + (1 - frac) * v2.z;
    return ret;
}

struct Vertex2D
{
    float x;
    float y;
    float z;
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
    Bitmap *frame_buffer;
    Mat4x4f *camera;
    float *camera_scale;

    Model *model;
    Mat4x4f *obj_coordinate;
    float *obj_scale;

    Mat3x3f rotate_;
    Vec3f move_;
    std::vector<Vec3f> vertex_;              // transformed
    std::vector<Vec3f> norms_;               // transformed
    std::vector<Vector<3, Vertex2D>> faces_; // clipped

    RenderObj(Render *render, Obj *obj);
    void render()
    {
        Mat4x4f transform = transformm_invert(*camera) * (*obj_coordinate); // 转换到相机空间
        rotate_ = transformm_rotate(transform);                             // 提取旋转矩阵
        move_ = transformm_move(transform);                                 // 提取位移

        // std::cout << "rotate_mm:\n"
        //           << rotate_ << "move_vec:\n"
        //           << move_ << std::endl;

// #pragma omp parallel for
        for (int i = 0; i < vertex_.size(); ++i)
        {
            vertex_[i] = rotate_ * (model->_verts[i]) * (*obj_scale) + move_; // 顶点坐标转换
            // std::cout << "vertex " << i << " after transfer\n"
            //           << vertex_[i] << std::endl;
        }
// #pragma omp parallel for
        for (int i = 0; i < norms_.size(); ++i)
        {
            norms_[i] = rotate_ * (model->_norms[i]); // 顶点法向量转换
            // std::cout << "norm " << i << " after transfer\n"
            //           << norms_[i] << std::endl;
        }
        clip_faces();
#pragma omp parallel for
        for (int i = 0; i < faces_.size();++i)
        {
            Draw_triangle(faces_[i]);
        }
    }
    void clip_faces()
    {
        faces_.clear();
        int mx = frame_buffer->_w / 2;
        int my = frame_buffer->_h / 2;

// #pragma omp parallel for
        for (int i = 0; i < model->_faces.size(); ++i)
        {
            Vec3i p1 = model->_faces[i][0]; // 顶点索引 / 纹理坐标索引 / 顶点法向量索引
            Vec3i p2 = model->_faces[i][1];
            Vec3i p3 = model->_faces[i][2];
            Vec3f v1 = vertex_[p1.x];
            Vec3f v2 = vertex_[p2.x];
            Vec3f v3 = vertex_[p3.x];
            if (v1.z > 0 && v2.z > 0 && v3.z > 0)
            {
                float z1 = v1.z / *camera_scale;
                float z2 = v2.z / *camera_scale;
                float z3 = v3.z / *camera_scale;
                Vertex2D vertex1 = {v1.x / z1 + mx, v1.y / z1 + my, z1, p1.y, p1.z};
                Vertex2D vertex2 = {v2.x / z2 + mx, v2.y / z2 + my, z2, p2.y, p2.z};
                Vertex2D vertex3 = {v3.x / z3 + mx, v3.y / z3 + my, z3, p3.y, p3.z};
                // sort p1, p2, p3
                if (vertex1.x > vertex2.x)
                    std::swap(vertex1, vertex2);
                if (vertex1.x > vertex3.x)
                    std::swap(vertex1, vertex3);
                if (vertex2.x > vertex3.x)
                    std::swap(vertex2, vertex3);
// #pragma omp critical
                {
                    faces_.push_back({vertex1, vertex2, vertex3});
                }
            }
        }
    }
    // |slope| < 1
    // input: (x0,y0) ,(x1,y1) and x1 <= x2, slope <= 1
    // output: step through x, output y
    struct Bresenham
    {
        int dx;
        int dy;
        int D;
        int y_step = 1;
        int y, ret = 0;
        bool flip = 1;
        Bresenham(int x0, int y0, int x1, int y1)
        {
            dx = x1 - x0;
            dy = y1 - y0;
            if (dy < 0)
            {
                y_step = -1; // if k < 0, only change the y direction
                dy = -dy;    // dy = abs(dy)
            }
            if (dx >= dy)
                flip = 0;
            else
                std::swap(dx, dy);
            D = -dx;
            y = y0;
        }
        inline int step(bool UP)
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
                        break;
                    }
                }
                return UP ? ret : y - y_step;
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
    struct _Face
    {
        float coff11, coff12, coff21, coff22, x3, y3;
        float z1, z2, z3;
        Vec3f norm1, norm2, norm3;
        Vec2f uv1, uv2, uv3;
    };
    void Draw_triangle(Vector<3, Vertex2D> &face)
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

        int c = (y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1); // up, down, line
        if (c == 0)                                            // not a line
            return;

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

        Bresenham l1(x1, y1, x3, y3);
        Bresenham l2(x1, y1, x2, y2);
        Bresenham l3(x2, y2, x3, y3);
        // std::cout << "xxxyyy:  " << x1 << ' ' << x2 << ' ' << x3 << ' ' << y1 << ' ' << y2 << ' ' << y3 << " \n";
        if (x2 > x1)
        {
            for (int i = x1; i < x2; ++i)
            {
                // std::cout << "scanline: " << i << std::endl;
                Draw_scanline(i, l1.step(c > 0), l2.step(c < 0), &f);
            }
        }
        if (x3 > x2)
        {
            for (int i = x2; i < x3; ++i)
            {
                // std::cout << "scanline: " << i << std::endl;
                Draw_scanline(i, l1.step(c > 0), l3.step(c < 0), &f);
            }
        }
    }
    inline void Draw_scanline(int x, int y1, int y2, _Face *f)
    {
        if (y1 > y2)
            std::swap(y1, y2);
        int dy = y2 - y1;
        for (int y = y1; y <= y2; ++y)
        {
            // double frac1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) * 1.0 / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            // double frac2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) * 1.0 / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            // double frac3 = 1 - frac1 - frac2;
            //std::cout << "draw_pixel x=" << x << "\t y=" << y << std::endl;

            float frac1 = f->coff11 * (x - f->x3) + f->coff12 * (y - f->y3);
            float frac2 = f->coff21 * (x - f->x3) + f->coff22 * (y - f->y3);
            float frac3 = 1 - frac1 - frac2;
            // if (frac1 >= 0 && frac2 >= 0 && frac3 >= 0)
            {
                frac1 /= f->z1;
                frac2 /= f->z2;
                frac3 /= f->z3;
                float z_ = 1.0 / (frac1 + frac2 + frac3);
                Vec2f uv_pixel = (frac1 * f->uv1 + frac2 * f->uv2 + frac3 * f->uv3) * z_;
                uint32_t argb = model->diffuse(uv_pixel);
                argb = ((argb & 0x000000ff) << 16) | ((argb & 0x00ff0000) >> 16) | ((argb & 0xff00ff00));
                frame_buffer->SetPixel(x, y, z_, argb);
            }
            // double z_ = 1.0 / (frac1 / f->z1 + frac2 / f->z2 + frac3 / f->z3);
            // Vec2f uv_pixel = ((float)frac1 * f->uv1 / f->z1 + (float)frac2 * f->uv2 / f->z2 + (float)frac3 * f->uv3 / f->z3) * (float)z_;
            // std::cout << "fracs: " << frac1 << "  " << frac2 << "  " << frac3 << std::endl
            // std::cout << "uv_coord " << uv_pixel << " x:" << x << " y:" << y << std::endl;
            //printf("deepth z_ =%12.10f %12.10f %12.10f | %12.10f\n",p1->z, p2->z, p3->z,z_);
            // frame_buffer->SetPixel(x, y, z_, vector_to_color(model->diffuse(uv_pixel)));
        }
    }
};

class Render
{
public:
    Bitmap *frame_buffer;
    Mat4x4f camera = matrix_set_identity();
    float camera_scale = 1;

    std::vector<RenderObj> obj_renders;

    Render(Bitmap &bitmap) : frame_buffer(&bitmap) {}
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
    void Draw_line(int x0, int y0, int x1, int y1, uint32_t color)
    {
        bool flip = false;
        if (std::abs(x0 - x1) < std::abs(y0 - y1))
        { // if dy > dx, swap x and y
            std::swap(x0, y0);
            std::swap(x1, y1);
            flip = true;
        }
        if (x0 > x1)
        { //  if x0 > x1, swap the start and end
            std::swap(x0, x1);
            std::swap(y0, y1);
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
                frame_buffer->SetPixel(y, x, color);
            else
                frame_buffer->SetPixel(x, y, color);
            D = D + 2 * dy;
            if (D > 0)
            {
                y = y + y_step; // next y
                D = D - 2 * dx;
            }
        }
    }
    // input: (x0,y0) ,(x1,y1) and x1 <= x2, slope <= 1
    // output: step through x, output y
};

RenderObj::RenderObj(Render *render, Obj *obj)
{
    frame_buffer = render->frame_buffer;
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
