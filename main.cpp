#include <GLFW/glfw3.h>
#include "include/render.h"


#define SCR_WIDTH 3000
#define SCR_HEIGHT 2000
#define ZOOM 1


int main()
{

    Render r(SCR_WIDTH, SCR_HEIGHT);
    r.set_camera({{1, 0, 0, 0},
                  {0, 1, 0, 0},
                  {0, 0, 1, -6.0},
                  {0, 0, 0, 1}},
                 1600.0);

    Model model("res/diablo3_pose.obj");
    Mat4x4f pose1 = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Mat4x4f pose2 = { {1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
    Obj test1(&model, pose1, 2);
    Obj test2(&model, pose2, 1);

    r.add_obj(test1);
    r.add_obj(test2);
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

        r.fb.fill(0xffffffff);
        test1.coordinate = test1.coordinate * matrix_set_rotate(0, 1, 0, 0.05);
        test2.coordinate = test2.coordinate * matrix_set_rotate(0, 1, 0, -0.05);
        r.render();
        // std::cout << "===============================new frame ==========================\n";
        glPixelZoom(ZOOM, ZOOM);
        glDrawPixels(SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, r.fb.fb_);
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
    test.coordinate = test.coordinate * matrix_set_rotate(1, 1, 1, 2.3);

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
    r.add_obj(test);
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
        test.coordinate = test.coordinate * matrix_set_rotate(0, 1, 0, 0.01);
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

