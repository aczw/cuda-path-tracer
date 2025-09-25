#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "glslUtility.hpp"
#include "gui_data.hpp"
#include "image.h"
#include "path_tracer.h"
#include "scene.h"
#include "scene_structs.h"
#include "utilities.cuh"

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

static std::string start_time;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camera_changed = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 camera_position;
glm::vec3 ogLookAt;  // for recentering the camera

RenderState* render_state;
int curr_iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint display_image;

ImGuiIO* io = nullptr;
bool is_mouse_over_imgui = false;

// Forward declarations for window loop and interactivity
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_pos_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

std::string get_current_time() {
  time_t now;
  time(&now);
  char buf[sizeof("0000-00-00_00-00-00z")];
  strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));

  return std::string(buf);
}

void initTextures() {
  glGenTextures(1, &display_image);
  glBindTexture(GL_TEXTURE_2D, display_image);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
  GLfloat vertices[] = {
      -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
  };

  GLfloat texcoords[] = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

  GLushort indices[] = {0, 1, 3, 3, 1, 2};

  GLuint vertexBufferObjID[3];
  glGenBuffers(3, vertexBufferObjID);

  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(positionLocation);

  glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
  glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(texcoordsLocation);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
  const char* attribLocations[] = {"Position", "Texcoords"};
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;

  // glUseProgram(program);
  if ((location = glGetUniformLocation(program, "u_image")) != -1) {
    glUniform1i(location, 0);
  }

  return program;
}

void deletePBO(GLuint* pbo) {
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);

    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);

    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex) {
  glDeleteTextures(1, tex);
  *tex = (GLuint)NULL;
}

void cleanupCuda() {
  if (pbo) {
    deletePBO(&pbo);
  }
  if (display_image) {
    deleteTexture(&display_image);
  }
}

void initCuda() {
  cudaGLSetGLDevice(0);

  // Clean up on program exit
  atexit(cleanupCuda);
}

void initPBO() {
  // set up vertex data parameter
  int num_texels = width * height;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo);
}

void error_callback(int error, const char* description) {
  std::cerr << std::format("{}", description) << std::endl;
}

/// Initialize CUDA and GL components.
GLFWwindow* initialize_components(Scene* scene) {
  glfwSetErrorCallback(error_callback);

  if (!glfwInit()) {
    exit(EXIT_FAILURE);
  }

  GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Path Tracer", nullptr, nullptr);

  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window);
  glfwSetWindowUserPointer(window, static_cast<void*>(scene));

  glfwSetKeyCallback(window, key_callback);
  glfwSetCursorPosCallback(window, mouse_pos_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);

  // Set up GL context
  glewExperimental = GL_TRUE;

  if (glewInit() != GLEW_OK) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  std::cout << std::format("OpenGL version: {}\n",
                           reinterpret_cast<const char*>(glGetString(GL_VERSION)));

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  io = &ImGui::GetIO();
  ImGui::StyleColorsLight();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 120");

  // Initialize other stuff
  initVAO();
  initTextures();
  initCuda();
  initPBO();

  GLuint passthrough_prog = initShader();
  glUseProgram(passthrough_prog);
  glActiveTexture(GL_TEXTURE0);

  return window;
}

void free_components(GLFWwindow* window) {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}

void render_gui(GuiData* gui_data) {
  is_mouse_over_imgui = io->WantCaptureMouse;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Info & Configuration");
  {
    ImGui::Text("Depth: %d", gui_data->max_depth);

    float fps = io->Framerate;
    ImGui::Text("FPS: %.2f (%.2f ms)", fps, 1000.0f / fps);

    ImGui::Separator();

    ImGui::Checkbox("Sort paths by material", &gui_data->sort_paths_by_material);
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

void save_image() {
  float num_samples = curr_iteration;
  // output image file
  Image image(width, height);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int index = x + (y * width);
      glm::vec3 pix = render_state->image[index];
      image.set_pixel(width - 1 - x, y, glm::vec3(pix) / num_samples);
    }
  }

  std::string file_name =
      std::format("{}_{}_{}samples", render_state->image_name, start_time, num_samples);
  image.save_as_png(file_name);
  // img.saveHDR(filename);
}

void loop(GLFWwindow* window, GuiData* gui_data, Scene* scene) {
  Camera& camera = render_state->camera;
  std::array<char, 10> iter_str;

  PathTracer path_tracer(gui_data, scene);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    if (camera_changed) {
      curr_iteration = 0;

      camera_position.x = zoom * std::sin(phi) * std::sin(theta);
      camera_position.y = zoom * std::cos(theta);
      camera_position.z = zoom * std::cos(phi) * std::sin(theta);

      camera.view = -glm::normalize(camera_position);
      glm::vec3 v = camera.view;
      glm::vec3 u = glm::vec3(0, 1, 0);  // glm::normalize(cam.up);
      glm::vec3 r = glm::cross(v, u);
      camera.up = glm::cross(r, v);
      camera.right = r;

      camera_position += camera.look_at;
      camera.position = camera_position;

      camera_changed = false;
    }

    if (curr_iteration == 0) {
      path_tracer.free();
      path_tracer.initialize();
    }

    if (curr_iteration < render_state->total_iterations) {
      uchar4* pbo_dptr = nullptr;
      curr_iteration++;

      // Map OpenGL buffer object for writing from CUDA on a single GPU.
      // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
      cudaGLMapBufferObject(reinterpret_cast<void**>(&pbo_dptr), pbo);

      path_tracer.run_iteration(pbo_dptr, curr_iteration);

      // Unmap buffer object
      cudaGLUnmapBufferObject(pbo);
    } else {
      save_image();
      path_tracer.free();
      cudaDeviceReset();

      exit(EXIT_SUCCESS);
    }

    std::string title = std::format("CIS 5650 CUDA Path Tracer | {} iterations", curr_iteration);
    glfwSetWindowTitle(window, title.c_str());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, display_image);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glClear(GL_COLOR_BUFFER_BIT);

    // Binding GL_PIXEL_UNPACK_BUFFER back to default
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

    // Render ImGui Stuff
    render_gui(gui_data);

    glfwSwapBuffers(window);
  }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action != GLFW_PRESS) {
    return;
  }

  switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GL_TRUE);
      break;

    case GLFW_KEY_S:
      save_image();
      break;

    case GLFW_KEY_SPACE:
      camera_changed = true;
      render_state = &(static_cast<Scene*>(glfwGetWindowUserPointer(window))->state);
      render_state->camera.look_at = ogLookAt;
      break;
  }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
  if (is_mouse_over_imgui) {
    return;
  }

  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mouse_pos_callback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) {
    // Otherwise, clicking back into window causes re-start
    return;
  }

  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, std::numbers::pi));
    camera_changed = true;
  } else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camera_changed = true;
  } else if (middleMousePressed) {
    render_state = &(static_cast<Scene*>(glfwGetWindowUserPointer(window))->state);
    Camera& cam = render_state->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.look_at -= (float)(xpos - lastX) * right * 0.01f;
    cam.look_at += (float)(ypos - lastY) * forward * 0.01f;
    camera_changed = true;
  }

  lastX = xpos;
  lastY = ypos;
}

int main(int argc, char* argv[]) {
  start_time = get_current_time();

  if (argc < 2) {
    std::cerr << std::format("Usage: {} <scene_file.json>", argv[0]) << std::endl;
    return EXIT_FAILURE;
  }

  std::unique_ptr scene = std::make_unique<Scene>(argv[1]);
  std::unique_ptr gui_data = std::make_unique<GuiData>(GuiData{
      .max_depth = scene->state.trace_depth,
      .sort_paths_by_material = true,
  });

  // Set up camera stuff from loaded path tracer settings
  curr_iteration = 0;
  render_state = &scene->state;
  Camera& cam = render_state->camera;
  width = cam.resolution.x;
  height = cam.resolution.y;

  glm::vec3 view = cam.view;
  glm::vec3 up = cam.up;
  glm::vec3 right = glm::cross(view, up);
  up = glm::cross(right, view);

  camera_position = cam.position;

  // compute phi (horizontal) and theta (vertical) relative 3D axis
  // so, (0 0 1) is forward, (0 1 0) is up
  glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
  glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
  phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
  theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
  ogLookAt = cam.look_at;
  zoom = glm::length(cam.position - ogLookAt);

  GLFWwindow* window = initialize_components(scene.get());
  loop(window, gui_data.get(), scene.get());
  free_components(window);

  return EXIT_SUCCESS;
}
