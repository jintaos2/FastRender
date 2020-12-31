#include <GLFW/glfw3.h>
#include "include/render.h"

#define SCR_WIDTH 1920
#define SCR_HEIGHT 1080
#define ZOOM 1

int main()
{
    Render r(SCR_WIDTH, SCR_HEIGHT);
    r.set_camera({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, -7.0}, {0, 0, 0, 1}}, 1400.0);

    Model model("res/diablo3_pose.obj");

    std::vector<Obj *> test_objs;
    std::vector<int> xxx = {-6, -5, -4, -3, -2, 2, 3, 4, 5, 6};
    for (int i = 0; i < 3; ++i)
    {
        for (int j : xxx)
        {
            for (int k = 0; k < 7; ++k)
            {
                Mat4x4f pose = {{1, 0, 0, (float)(j)}, {0, 1, 0, (float)(i - 1) * 2}, {0, 0, 1, (float)k * 3 + 2}, {0, 0, 0, 1}};
                Obj *a = new Obj(&model, pose, 0.8 + 0.2 * (k + 1));
                test_objs.push_back(a);
            }
        }
    }


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

        for (int i = 0; i < test_objs.size(); ++i)
        {
            test_objs[i]->coordinate = test_objs[i]->coordinate * matrix_set_rotate(i, 1, -i, 0.05 - i / 300.0);
        }
        r.render(0xffffffff);
        std::cout << ">> FPS: ";
        if (count_FPS == 10)
        {
            std::cout << 10 / (glfwGetTime() - glfw_time);
            count_FPS = 0;
            glfw_time = glfwGetTime();
        }
        std::cout << std::endl;
        glPixelZoom(ZOOM, ZOOM);
        glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, r.get_framebuffer());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
