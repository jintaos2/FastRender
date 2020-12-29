#include <GLFW/glfw3.h>
#include "include/render.h"

#define SCR_WIDTH 1920
#define SCR_HEIGHT 1080
#define ZOOM 1

int main()
{
    Render r(SCR_WIDTH, SCR_HEIGHT);
    r.set_camera({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, -7.0}, {0, 0, 0, 1}}, 1400.0);

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
    Model mymodel("xxx");
    mymodel._diffusemap = &mybitmap;
    mymodel._verts.push_back({-300, -300, -1});
    mymodel._verts.push_back({300, -300, -1});
    mymodel._verts.push_back({-300, -300, 1});
    mymodel._verts.push_back({-300, 300, -1});
    mymodel._faces.push_back({{0, 2, 0}, {1, 2, 0}, {2, 2, 0}});
    mymodel._faces.push_back({{0, 1, 0}, {1, 1, 0}, {3, 1, 0}});
    mymodel._faces.push_back({{0, 0, 0}, {2, 0, 0}, {3, 0, 0}});
    mymodel._faces.push_back({{1, 3, 0}, {2, 3, 0}, {3, 3, 0}});
    mymodel._norms.push_back({1, 1, 1});
    mymodel._uv.push_back({0.01, 0.01});
    mymodel._uv.push_back({0.01, 0.99});
    mymodel._uv.push_back({0.99, 0.01});
    mymodel._uv.push_back({0.99, 0.99});

    Obj test3(&mymodel, {{1, 0, 0, 0.3}, {0, 1, 0, 0.6}, {0, 0, 1, -5}, {0, 0, 0, 1}}, 0.001);

    Model model("res/diablo3_pose.obj");
    Obj test1(&model, {{1, 0, 0, -1}, {0, 1, 0, 0}, {0, 0, 1, -2}, {0, 0, 0, 1}}, 1);
    Obj test2(&model, {{1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 1, -2}, {0, 0, 0, 1}}, 1);

    std::vector<Obj *> test_objs;
    for (int k = 0; k < 45; ++k)
    {
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 9; ++j)
            {
                Mat4x4f pose = {{1, 0, 0, (float)(j - 4)}, {0, 1, 0, (float)(i - 2)}, {0, 0, 1, (float)k * 3}, {0, 0, 0, 1}};
                Obj *a = new Obj(&model, pose, 0.6 + 0.6 * (k + 1));
                test_objs.push_back(a);
            }
        }
    }

    r.add_obj(&test1);
    r.add_obj(&test2);
    r.add_obj(&test3);

    for (int i = 0; i < test_objs.size(); ++i)
    {
        r.add_obj(test_objs[i]);
    }

    ////////////////////
    // GLFW
    ////////////////////
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
            r.rotate_camera_up(0.05);
        else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            r.rotate_camera_up(-0.05);
        else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            r.rotate_camera_left(0.05);
        else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            r.rotate_camera_left(-0.05);

        r.fb.fill(0xffffffff);
        test1.coordinate = test1.coordinate * matrix_set_rotate(0, 1, 0, 0.05);
        test2.coordinate = test2.coordinate * matrix_set_rotate(0, 1, 0, -0.05);
        test3.coordinate = test3.coordinate * matrix_set_rotate(0, 0, 1, -0.02);

        for (int i = 0; i < test_objs.size(); ++i)
        {
            test_objs[i]->coordinate = test_objs[i]->coordinate * matrix_set_rotate(i, 1, -i, 0.05 - i / 300.0);
        }
        r.render();
        // std::cout << "===============================new frame ==========================\n";
        glPixelZoom(ZOOM, ZOOM);
        glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, r.fb.fb_);
        glfwSwapBuffers(window);
        glfwPollEvents();
        // break;
    }
    glfwTerminate();
    return 0;
}
