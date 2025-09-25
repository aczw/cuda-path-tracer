#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "glslUtility.hpp"
#include "gui_data.hpp"
#include "image.hpp"
#include "path_tracer.h"
#include "render_context.hpp"
#include "scene.hpp"
#include "scene_structs.h"
#include "utilities.cuh"
#include "window.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint display_image;

void initialize_textures(int width, int height) {
  glGenTextures(1, &display_image);
  glBindTexture(GL_TEXTURE_2D, display_image);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initialize_vao(void) {
  GLfloat vertices[] = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
  GLfloat uv[] = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  GLushort indices[] = {0, 1, 3, 3, 1, 2};

  GLuint vertexBufferObjID[3];
  glGenBuffers(3, vertexBufferObjID);

  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(positionLocation);

  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uv), uv, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(texcoordsLocation);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initialize_shader() {
  const char* attribLocations[] = {"Position", "Texcoords"};
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;

  // glUseProgram(program);
  if ((location = glGetUniformLocation(program, "u_image")) != -1) {
    glUniform1i(location, 0);
  }

  return program;
}

void initialize_pbo(int num_pixels) {
  // Set up vertex data parameter
  int num_values = num_pixels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo);
}

/// Initialize CUDA, OpenGL, and GUI components.
bool initialize_components(RenderContext* ctx, GLFWwindow* window) {
  // Set up GL context
  glewExperimental = GL_TRUE;

  if (glewInit() != GLEW_OK) {
    std::cerr << "[GLEW] Failed to initialize" << std::endl;
    return false;
  }

  std::cout << std::format("[Info] OpenGL version: {}\n",
                           reinterpret_cast<const char*>(glGetString(GL_VERSION)));

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsLight();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 120");

  int width = ctx->get_width();
  int height = ctx->get_height();

  initialize_vao();
  initialize_textures(width, height);
  // cudaGLSetGLDevice(0);
  initialize_pbo(width * height);

  glUseProgram(initialize_shader());
  glActiveTexture(GL_TEXTURE0);

  return true;
}

void free_components() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  if (pbo) {
    // Unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(pbo);

    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glDeleteBuffers(1, &pbo);

    pbo = (GLuint)NULL;
  }

  if (display_image) {
    glDeleteTextures(1, &display_image);
    display_image = (GLuint)NULL;
  }
}

void render_gui(GuiData* gui_data) {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Info & Configuration");
  {
    ImGui::Text("Depth: %d", gui_data->max_depth);

    float fps = ImGui::GetIO().Framerate;
    ImGui::Text("FPS: %.2f (%.2f ms)", fps, 1000.0f / fps);

    ImGui::Separator();

    ImGui::Checkbox("Sort paths by material", &gui_data->sort_paths_by_material);
    ImGui::Checkbox("Discard paths that went out of bounds", &gui_data->discard_oob_paths);
    ImGui::Checkbox("Discard paths that intersected with a light",
                    &gui_data->discard_light_isect_paths);
    ImGui::Checkbox("Stochastic sampling", &gui_data->stochastic_sampling);
  }
  ImGui::End();

  // ImGui::Text("This is some useful text.");
  // ImGui::Checkbox("Demo Window", &show_demo_window);
  // ImGui::Checkbox("Another Window", &show_another_window);

  // ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
  // ImGui::ColorEdit3("clear color", (float*)&clear_color);

  // if (ImGui::Button("Button")) {
  //   counter++;
  // }

  // ImGui::SameLine();
  // ImGui::Text("counter = %d", counter);

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void loop(RenderContext* ctx, GLFWwindow* window, GuiData* gui_data) {
  std::array<char, 10> iter_str;
  PathTracer path_tracer(ctx, gui_data);

  float prev_zoom, prev_theta, prev_phi;
  Camera prev_camera = ctx->scene.camera;
  GuiData prev_gui_data = *gui_data;

  int width = ctx->get_width();
  int height = ctx->get_height();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    if (prev_camera != ctx->scene.camera || prev_zoom != ctx->zoom || prev_theta != ctx->theta ||
        prev_phi != ctx->phi) {
      ctx->curr_iteration = 0;
      ctx->scene.camera.update(ctx->zoom, ctx->theta, ctx->phi);

      prev_zoom = ctx->zoom;
      prev_theta = ctx->theta;
      prev_phi = ctx->phi;
      prev_camera = ctx->scene.camera;
    }

    if (prev_gui_data.stochastic_sampling != gui_data->stochastic_sampling) {
      ctx->curr_iteration = 0;
    }

    if (ctx->curr_iteration == 0) {
      path_tracer.free();
      path_tracer.initialize();
    }

    if (ctx->curr_iteration < ctx->settings.max_iterations) {
      uchar4* pbo_dptr = nullptr;
      ctx->curr_iteration += 1;

      // Map OpenGL buffer object for writing from CUDA on a single GPU.
      // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
      cudaGLMapBufferObject(reinterpret_cast<void**>(&pbo_dptr), pbo);

      path_tracer.run_iteration(pbo_dptr, ctx->curr_iteration);

      // Unmap buffer object
      cudaGLUnmapBufferObject(pbo);
    } else {
      ctx->save_image();
      path_tracer.free();
      cudaDeviceReset();

      return;
    }

    std::string title =
        std::format("CIS 5650 CUDA Path Tracer | {} iterations", ctx->curr_iteration);
    glfwSetWindowTitle(window, title.c_str());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, display_image);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glClear(GL_COLOR_BUFFER_BIT);

    // Binding GL_PIXEL_UNPACK_BUFFER back to default
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

    prev_gui_data = *gui_data;
    ctx->input.mouse_over_gui = ImGui::GetIO().WantCaptureMouse;
    render_gui(gui_data);

    glfwSwapBuffers(window);
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << std::format("Usage: {} <scene_file.json>", argv[0]) << std::endl;
    return EXIT_FAILURE;
  }

  std::unique_ptr ctx = std::make_unique<RenderContext>();

  if (!ctx->try_open_scene(argv[1])) {
    return EXIT_FAILURE;
  }

  std::unique_ptr gui_data = std::make_unique<GuiData>(GuiData{
      .max_depth = ctx->settings.max_depth,
      .sort_paths_by_material = true,
      .discard_oob_paths = true,
      .discard_light_isect_paths = true,
      .stochastic_sampling = true,
  });

  Window window(ctx.get());

  if (!window.try_init()) {
    return EXIT_FAILURE;
  }

  if (!initialize_components(ctx.get(), window.get())) {
    return EXIT_FAILURE;
  }

  loop(ctx.get(), window.get(), gui_data.get());
  free_components();

  return EXIT_SUCCESS;
}
