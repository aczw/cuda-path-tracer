#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "glslUtility.hpp"
#include "gui_data.hpp"
#include "image.hpp"
#include "path_segment.hpp"
#include "path_tracer.hpp"
#include "render_context.hpp"
#include "scene.hpp"
#include "utilities.cuh"
#include "window.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <tiny_gltf.h>

#include <array>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>

void initialize_textures(RenderContext* ctx, int width, int height) {
  glGenTextures(1, &ctx->display_image);
  glBindTexture(GL_TEXTURE_2D, ctx->display_image);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initialize_vao(void) {
  GLfloat vertices[] = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
  GLfloat uv[] = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  GLushort indices[] = {0, 1, 3, 3, 1, 2};

  GLuint vertex_buffer_obj_id[3];
  glGenBuffers(3, vertex_buffer_obj_id);

  GLuint position_location = 0;
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_id[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)position_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(position_location);

  GLuint uv_location = 1;
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_id[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uv), uv, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)uv_location, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(uv_location);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_buffer_obj_id[2]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initialize_shader() {
  const char* attribLocations[] = {"Position", "Texcoords"};
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;

  if ((location = glGetUniformLocation(program, "u_image")) != -1) {
    glUniform1i(location, 0);
  }

  return program;
}

void initialize_pbo(RenderContext* ctx, int num_pixels) {
  // Set up vertex data parameter
  int num_values = num_pixels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &ctx->pbo);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ctx->pbo);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(ctx->pbo);
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
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 120");

  int width = ctx->get_width();
  int height = ctx->get_height();

  initialize_vao();
  initialize_textures(ctx, width, height);
  initialize_pbo(ctx, width * height);

  glUseProgram(initialize_shader());
  glActiveTexture(GL_TEXTURE0);

  return true;
}

void free_components(RenderContext* ctx) {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  if (ctx->pbo) {
    // Unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(ctx->pbo);

    glBindBuffer(GL_ARRAY_BUFFER, ctx->pbo);
    glDeleteBuffers(1, &ctx->pbo);

    ctx->pbo = 0;
  }

  if (ctx->display_image) {
    glDeleteTextures(1, &ctx->display_image);
    ctx->display_image = 0;
  }
}

void render_gui(GuiData* gui_data) {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Info & Configuration");
  {
    Settings* settings = gui_data->settings;
    ImGui::Text("Scene: \"%s\"", settings->scene_name.c_str());
    ImGui::Text("Max iterations: %d", settings->max_iterations);
    ImGui::Text("Max depth: %d", settings->max_depth);

    float fps = ImGui::GetIO().Framerate;
    ImGui::Text("FPS: %.2f (%.2f ms)", fps, 1000.0f / fps);

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::BeginTabBar("Configuration")) {
      if (ImGui::BeginTabItem("Performance")) {
        ImGui::Checkbox("Sort paths by material", &gui_data->sort_paths_by_material);

        {
          ImGui::Text("Discard paths that:");
          ImGui::Checkbox("Traveled out of bounds", &gui_data->discard_oob_paths);
          ImGui::Checkbox("Intersected with a light", &gui_data->discard_light_isect_paths);
        }

        ImGui::EndTabItem();
      }

      if (ImGui::BeginTabItem("Visual")) {
        ImGui::Checkbox("Apply tone mapping", &gui_data->apply_tone_mapping);
        ImGui::EndTabItem();
      }

      if (ImGui::BeginTabItem("Camera")) {
        ImGui::Checkbox("Perform stochastic sampling", &gui_data->camera.stochastic_sampling);

        ImGui::Checkbox("Enable depth of field", &gui_data->camera.depth_of_field);
        {
          const bool enabled = gui_data->camera.depth_of_field;
          if (!enabled) ImGui::BeginDisabled();
          ImGui::PushItemWidth(150.f);
          ImGui::SliderFloat("Lens radius", &gui_data->camera.lens_radius, 0.f, 10.f);
          ImGui::SliderFloat("Focal distance", &gui_data->camera.focal_distance, 0.f, 50.f);
          ImGui::PopItemWidth();
          if (!enabled) ImGui::EndDisabled();
        }

        ImGui::EndTabItem();
      }
    }

    ImGui::EndTabBar();
  }
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void loop(RenderContext* ctx, GLFWwindow* window) {
  std::array<char, 10> iter_str;
  GuiData* gui_data = ctx->get_gui_data();
  PathTracer path_tracer(ctx);

  float prev_zoom = -1.f, prev_theta = -1.f, prev_phi = -1.f;
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

    if (prev_gui_data.camera != gui_data->camera ||
        prev_gui_data.apply_tone_mapping != gui_data->apply_tone_mapping) {
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
      cudaGLMapBufferObject(reinterpret_cast<void**>(&pbo_dptr), ctx->pbo);

      path_tracer.run_iteration(pbo_dptr, ctx->curr_iteration);

      // Unmap buffer object
      cudaGLUnmapBufferObject(ctx->pbo);
    } else {
      ctx->save_image();
      path_tracer.free();
      cudaDeviceReset();

      return;
    }

    std::string title =
        std::format("CIS 5650 CUDA Path Tracer | {} iterations", ctx->curr_iteration);
    glfwSetWindowTitle(window, title.c_str());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ctx->pbo);
    glBindTexture(GL_TEXTURE_2D, ctx->display_image);
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

  Window window(ctx.get());

  if (!window.try_init()) {
    return EXIT_FAILURE;
  }

  if (!initialize_components(ctx.get(), window.get())) {
    return EXIT_FAILURE;
  }

  using namespace tinygltf;

  Model model;
  TinyGLTF loader;
  std::string error;
  std::string warning;

  bool result = loader.LoadBinaryFromFile(&model, &error, &warning, "../../../models/cube.glb");

  if (!warning.empty()) {
    std::cout << std::format("[GLTF] Warning: {}\n", warning);
  }

  if (!error.empty()) {
    std::cout << std::format("[GLSL] Error: {}\n", error);
  }

  if (!result) {
    return EXIT_FAILURE;
  }

  loop(ctx.get(), window.get());
  free_components(ctx.get());

  return EXIT_SUCCESS;
}
