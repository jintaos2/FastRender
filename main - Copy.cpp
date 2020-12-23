#include <GLFW/glfw3.h>
#include <stdio.h>
#include "include/bitmap.h"
#include "include/shader.h"
#include "include/model.h"

#define SCR_WIDTH 600
#define SCR_HEIGHT 400
#define ZOOM 2

struct Vertex3D
{
    Vec3f v;
    Vec3f norm;
    Vec2f uv;
};
struct Vertex2D
{
    int x;
    int y;
    float z;
    Vec3f norm;
    Vec2f uv;
};
inline std::ostream &operator<<(std::ostream &os, const Vertex2D &a)
{
    os << "x:";
    os << a.x << " y:" << a.y << " z:" << a.z << "  norm:" << a.norm << "  uv_coord:" << a.uv;
    return os;
}

// template <typename T>
// inline static T interp(T v1, T v2, float frac)
// {
//     T ret = v1 * frac + v2 * (1 - frac);
//     return ret;
// }
inline static Vec3d interp(Vec3f v1, Vec3f v2, double frac)
{
    Vec3d ret;
    ret.x = v1.x * frac + (1 - frac) * v2.x;
    ret.y = v1.y * frac + (1 - frac) * v2.y;
    ret.z = v1.z * frac + (1 - frac) * v2.z;
    return ret;
}

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
    std::vector<Vec3f> vertex_; // transformed
    std::vector<Vec3f> norms_;  // transformed
    std::vector<Vector<3, Vertex2D>> faces_;  // clipped

    RenderObj(Obj *obj_, Bitmap *fb_, Mat4x4f *camera_, float *camera_scale_) : frame_buffer(fb_), camera(camera_), camera_scale(camera_scale_)
    {
        model = obj_->model;
        obj_coordinate = &(obj_->coordinate);
        obj_scale = &(obj_->scale);
        vertex_.resize(model->_verts.size());
        norms_.resize(model->_norms.size());
        faces_.reserve(model->_faces.size());
    }
    void render()
    {
        Mat4x4f transform = transformm_invert(*camera) * (*obj_coordinate);
        rotate_ = transformm_rotate(transform);
        move_ = transformm_move(transform);
        // std::cout << "rotate_mm:\n"
        //           << rotate_ << "move_vec:\n"
        //           << move_ << std::endl;
        float s = *obj_scale;
        for (int i = 0; i < vertex_.size(); ++i)
        {
            vertex_[i] = rotate_ * (model->_verts[i]) * s + move_; // 顶点坐标转换
            // std::cout << "vertex " << i << " after transfer\n"
            //           << vertex_[i] << std::endl;
        }
        for (int i = 0; i < norms_.size(); ++i)
        {
            norms_[i] = rotate_ * (model->_norms[i]); // 顶点法向量转换
            // std::cout << "norm " << i << " after transfer\n"
            //           << norms_[i] << std::endl;
        }
        clip_faces();
        for (auto i : faces_)
        {
            Draw_triangle(i);
        }
    }
    void clip_faces()
    {
        faces_.clear();
        int middle_x = frame_buffer->GetW() / 2;
        int middle_y = frame_buffer->GetH() / 2;
        for (auto i : model->_faces)
        {
            Vec3i p1 = i[0]; // 顶点索引 / 纹理坐标索引 / 顶点法向量索引
            Vec3i p2 = i[1];
            Vec3i p3 = i[2];
            Vec3f v1 = vertex_[p1.x];
            Vec3f v2 = vertex_[p2.x];
            Vec3f v3 = vertex_[p3.x];
            if (v1.z > 0 && v2.z > 0 && v3.z > 0)
            {
                float z1 = v1.z / *camera_scale;
                float z2 = v2.z / *camera_scale;
                float z3 = v3.z / *camera_scale;
                int x1 = round(v1.x / z1) + middle_x;
                int x2 = round(v2.x / z2) + middle_x;
                int x3 = round(v3.x / z3) + middle_x;
                int y1 = round(v1.y / z1) + middle_y;
                int y2 = round(v2.y / z2) + middle_y;
                int y3 = round(v3.y / z3) + middle_y;
                faces_.push_back({{x1, y1, z1, norms_[p1.z], model->_uv[p1.y]},
                                  {x2, y2, z2, norms_[p2.z], model->_uv[p2.y]},
                                  {x3, y3, z3, norms_[p3.z], model->_uv[p3.y]}});
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

    void Draw_triangle(Vector<3, Vertex2D> &face)
    {
        Vertex2D *p1 = &(face.x);
        Vertex2D *p2 = &(face.y);
        Vertex2D *p3 = &(face.z);
        if (p1->x > p2->x)
            std::swap(p1, p2); // sort p1, p2
        if (p1->x > p3->x)
            std::swap(p1, p3); // p1-> is minimum
        if (p2->x > p3->x)
            std::swap(p2, p3);

        // std::cout << "-------------------Draw_triangle----------------------: \n"
        //           << *p1 << "\n"
        //           << *p2 << "\n"
        //           << *p3 << '\n';

        int c = (p3->y - p1->y) * (p2->x - p1->x) - (p2->y - p1->y) * (p3->x - p1->x); // up, down, line
        if (c == 0)
            return;
        Bresenham l1(p1->x, p1->y, p3->x, p3->y);
        Bresenham l2(p1->x, p1->y, p2->x, p2->y);
        Bresenham l3(p2->x, p2->y, p3->x, p3->y);
        if (p2->x > p1->x)
        {
            for (int i = p1->x; i < p2->x; ++i)
            {
                // float frac13 = (float)(i - p1->x)  / (p3->x - p1->x);
                // float frac12 = (float)(i - p1->x)  / (p2->x - p1->x);
                Draw_scanline(i, l1.step(c > 0), l2.step(c < 0), p1, p2, p3);
            }
        }
        if (p3->x > p2->x)
        {
            for (int i = p2->x; i < p3->x; ++i)
            {
                // float frac13 = (float)(i - p1->x) / (p3->x - p1->x);
                // float frac23 = (float)(i - p2->x) / (p3->x - p2->x);
                Draw_scanline(i, l1.step(c > 0), l3.step(c < 0), p1, p2, p3);
            }
        }
    }
    inline void Draw_scanline(int x, int y1, int y2, Vertex2D *p1, Vertex2D *p2, Vertex2D *p3)
    {
        if (y1 > y2)
            std::swap(y1, y2);
        int dy = y2 - y1;
        for (int y = y1; y <= y2; ++y)
        {
            int x1 = p1->x;
            int y1 = p1->y;
            int x2 = p2->x;
            int y2 = p2->y;
            int x3 = p3->x;
            int y3 = p3->y;

            double frac1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) * 1.0 / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            double frac2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) * 1.0 / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            double frac3 = 1 - frac1 - frac2;
            //std::cout << interp_coff << p1->z << "  " << p2->z << "  " << p3->z << std::endl;
            double z_ = 1.0 / (frac1 / p1->z + frac2 / p2->z + frac3 / p3->z);
            Vec2f uv_pixel = ((float)frac1 * p1->uv / p1->z + (float)frac2 * p2->uv / p2->z + (float)frac3 * p3->uv / p3->z) * (float)z_;
            //printf("deepth z_ =%12.10f %12.10f %12.10f | %12.10f\n",p1->z, p2->z, p3->z,z_);
            frame_buffer->SetPixel(x, y, z_, vector_to_color(model->diffuse(uv_pixel)));
        }
    }
};

struct Obj
{
public:
    Model *model;
    Mat4x4f coordinate = matrix_set_identity();
    float scale = 1;
    Obj(Model *model_, Mat4x4f pose_, float scale_) : model(model_), coordinate(pose_), scale(scale_) {}
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
        obj_renders.push_back(RenderObj(&obj, frame_buffer, &camera, &camera_scale));
    }
    void render()
    {
        for (auto i : objs)
        {
            RenderObj obj(frame_buffer, i->model, &(i->pose), &camera, &(i->scale));
            obj.render(camera_scale);
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

int main()
{

    Bitmap frame_buffer_(SCR_WIDTH, SCR_HEIGHT);
    Render r(frame_buffer_);
    r.set_camera({{1, 0, 0, 0},
                  {0, 1, 0, 0},
                  {0, 0, 1, -2.0},
                  {0, 0, 0, 1}},
                 200.0);

    Model model("res/diablo3_pose.obj");
    Mat4x4f pose = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Obj test(&model, pose, 1);

    r.objs.push_back(&test);
    // r.render();

    // r.Draw_triangle3D(aa, bb, cc, camera_dir, camera_pos);
    // r.Draw_triangle(&a, &b, &c, 0);

    glfwInit();
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH * ZOOM, SCR_HEIGHT * ZOOM, "test", NULL, NULL);
    glfwMakeContextCurrent(window);
    int count_FPS = 0;
    double glfw_time = glfwGetTime();
    while (!glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        count_FPS++;
        if (count_FPS == 10)
        {
            std::cout << 10 / (glfwGetTime() - glfw_time) << " FPS\n";
            count_FPS = 0;
            glfw_time = glfwGetTime();
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            r.move_camera_y(0.1);
        else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            r.move_camera_y(-0.1);
        else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            r.move_camera_x(-0.1);
        else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            r.move_camera_x(0.1);
        else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            r.rotate_camera_up(0.1);
        else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            r.rotate_camera_up(-0.1);
        else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            r.rotate_camera_left(-0.1);
        else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            r.rotate_camera_left(0.1);

        frame_buffer_.Fill(0xffffffff);
        test.pose = test.pose * matrix_set_rotate(0, 1, 0, 0.05);
        r.render();
        // std::cout << "===============================new frame ==========================\n";
        glPixelZoom(ZOOM, ZOOM);
        glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, frame_buffer_.GetBits());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    // frame_buffer_.SaveFile("output.bmp");
    return 0;
}

/*
int main()
{

    Bitmap frame_buffer_(SCR_WIDTH, SCR_HEIGHT);
    Render r(frame_buffer_);
    r.set_camera({{1, 0, 0, 0},
                  {0, 1, 0, 0},
                  {0, 0, 1, -1200.0},
                  {0, 0, 0, 1}},
                 200.0);

    // Model model("res/diablo3_pose.obj");
    Model mymodel("test");
    Mat4x4f pose = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Obj test(&mymodel, pose, 1);
    test.pose = test.pose * matrix_set_rotate(1, 1, 1, 2.3);

    Bitmap mybitmap(4, 4);
    mybitmap.SetPixel(0, 0, 0xff0000ff);
    mybitmap.SetPixel(0, 1, 0xff0000ff);
    mybitmap.SetPixel(1, 0, 0xff0000ff);
    mybitmap.SetPixel(1, 1, 0xff0000ff);
    mybitmap.SetPixel(2, 0, 0xff00ff00);
    mybitmap.SetPixel(2, 1, 0xff00ff00);
    mybitmap.SetPixel(3, 0, 0xff00ff00);
    mybitmap.SetPixel(3, 1, 0xff00ff00);
    mybitmap.SetPixel(0, 2, 0xffff0000);
    mybitmap.SetPixel(0, 3, 0xffff0000);
    mybitmap.SetPixel(1, 2, 0xffff0000);
    mybitmap.SetPixel(1, 3, 0xffff0000);
    mybitmap.SetPixel(2, 2, 0xff000000);
    mybitmap.SetPixel(2, 3, 0xff000000);
    mybitmap.SetPixel(3, 2, 0xff000000);
    mybitmap.SetPixel(3, 3, 0xff000000);
    test.model->_diffusemap = &mybitmap;
    test.model->_verts.push_back({-300, -300, -300});
    test.model->_verts.push_back({300, -300, -300});
    test.model->_verts.push_back({-300, -300, 300});
    test.model->_verts.push_back({-300, 300, -300});
    test.model->_faces.push_back({{0, 2, 0}, {1, 2, 0}, {2, 2, 0}});
    test.model->_faces.push_back({{0, 1, 0}, {1, 1, 0}, {3, 1, 0}});
    test.model->_faces.push_back({{0, 0, 0}, {2, 0, 0}, {3, 0, 0}});
    test.model->_faces.push_back({{1, 3, 0}, {2, 3, 0}, {3, 3, 0}});
    test.model->_norms.push_back({1, 1, 1});
    test.model->_uv.push_back({0.01, 0.01});
    test.model->_uv.push_back({0.01, 0.99});
    test.model->_uv.push_back({0.99, 0.01});
    test.model->_uv.push_back({0.99, 0.99});

    r.objs.push_back(&test);
    // r.render();

    // r.Draw_triangle3D(aa, bb, cc, camera_dir, camera_pos);
    // r.Draw_triangle(&a, &b, &c, 0);

  
    glfwInit();
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH * ZOOM, SCR_HEIGHT * ZOOM, "test", NULL, NULL);
    glfwMakeContextCurrent(window);
    int count_FPS = 0;
    double glfw_time = glfwGetTime();
    while (!glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        count_FPS++;
        if (count_FPS == 10)
        {
            std::cout << 10 / (glfwGetTime() - glfw_time) << " FPS\n";
            count_FPS = 0;
            glfw_time = glfwGetTime();
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            r.move_camera_y(10);
        else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            r.move_camera_y(-10);
        else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            r.move_camera_x(-10);
        else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            r.move_camera_x(10);
        else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            r.rotate_camera_up(0.1);
        else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            r.rotate_camera_up(-0.1);
        else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            r.rotate_camera_left(-0.1);
        else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            r.rotate_camera_left(0.1);

        frame_buffer_.Fill(0xffffffff);
        test.pose = test.pose * matrix_set_rotate(0, 1, 0, 0.01);
        r.render();
        // std::cout << "===============================new frame ==========================\n";
        glPixelZoom(ZOOM, ZOOM);
        glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, frame_buffer_.GetBits());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    // frame_buffer_.SaveFile("output.bmp");
    return 0;
}

*/