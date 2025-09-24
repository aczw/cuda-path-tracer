#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "glslUtility.hpp"
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
#include <glm/gtx/transform.hpp>

#include <array>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
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

Scene* scene;
GuiDataContainer* gui_data;
RenderState* render_state;
int curr_iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Forward declarations for window loop and interactivity
void run_cuda();
void keyCallback(GLFWwindow* window,
                 int key,
                 int scancode,
                 int action,
                 int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

std::string get_current_time() {
  time_t now;
  time(&now);
  char buf[sizeof("0000-00-00_00-00-00z")];
  strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));

  return std::string(buf);
}

void initTextures() {
  glGenTextures(1, &displayImage);
  glBindTexture(GL_TEXTURE_2D, displayImage);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
               GL_UNSIGNED_BYTE, NULL);
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
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);
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
  if (displayImage) {
    deleteTexture(&displayImage);
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

void errorCallback(int error, const char* description) {
  fprintf(stderr, "%s\n", description);
}

bool init() {
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    exit(EXIT_FAILURE);
  }

  window = glfwCreateWindow(width, height, "CUDA Path Tracer", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  // Set up GL context
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }
  printf("OpenGL version: %s\n", glGetString(GL_VERSION));

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
  GLuint passthroughProgram = initShader();

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  return true;
}

void InitImguiData(GuiDataContainer* gui_data) {
  imguiData = gui_data;
}

// LOOK: Un-Comment to check ImGui Usage
void RenderImGui() {
  mouseOverImGuiWinow = io->WantCaptureMouse;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  bool show_demo_window = true;
  bool show_another_window = false;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  static float f = 0.0f;
  static int counter = 0;

  ImGui::Begin("Path Tracer Analytics");  // Create a window called "Hello,
                                          // world!" and append into it.

  // LOOK: Un-Comment to check the output window and usage
  // ImGui::Text("This is some useful text.");               // Display some
  // text (you can use a format strings too) ImGui::Checkbox("Demo Window",
  // &show_demo_window);      // Edit bools storing our window open/close state
  // ImGui::Checkbox("Another Window", &show_another_window);

  // ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float
  // using a slider from 0.0f to 1.0f ImGui::ColorEdit3("clear color",
  // (float*)&clear_color); // Edit 3 floats representing a color

  // if (ImGui::Button("Button"))                            // Buttons return
  // true when clicked (most widgets return true when edited/activated)
  //     counter++;
  // ImGui::SameLine();
  // ImGui::Text("counter = %d", counter);
  ImGui::Text("Traced Depth %d", imguiData->traced_depth);
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
              1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool MouseOverImGuiWindow() {
  return mouseOverImGuiWinow;
}

void run_main_loop() {
  std::array<char, 10> iter_str;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    run_cuda();

    std::string title = std::format("CIS 5650 CUDA Path Tracer | {} iterations",
                                    curr_iteration);
    glfwSetWindowTitle(window, title.c_str());

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                    GL_UNSIGNED_BYTE, NULL);
    glClear(GL_COLOR_BUFFER_BIT);

    // Binding GL_PIXEL_UNPACK_BUFFER back to default
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

    // Render ImGui Stuff
    RenderImGui();

    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}

void saveImage() {
  float samples = curr_iteration;
  // output image file
  Image img(width, height);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int index = x + (y * width);
      glm::vec3 pix = render_state->image[index];
      img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
    }
  }

  std::string filename = render_state->image_name;
  std::ostringstream ss;
  ss << filename << "." << start_time << "." << samples << "samp";
  filename = ss.str();

  // CHECKITOUT
  img.savePNG(filename);
  // img.saveHDR(filename);  // Save a Radiance HDR file
}

void run_cuda() {
  if (camera_changed) {
    curr_iteration = 0;
    Camera& camera = render_state->camera;
    camera_position.x = zoom * sin(phi) * sin(theta);
    camera_position.y = zoom * cos(theta);
    camera_position.z = zoom * cos(phi) * sin(theta);

    camera.view = -glm::normalize(camera_position);
    glm::vec3 v = camera.view;
    glm::vec3 u = glm::vec3(0, 1, 0);  // glm::normalize(cam.up);
    glm::vec3 r = glm::cross(v, u);
    camera.up = glm::cross(r, v);
    camera.right = r;

    camera.position = camera_position;
    camera_position += camera.look_at;

    camera.position = camera_position;
    camera_changed = false;
  }

  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use
  // this buffer

  if (curr_iteration == 0) {
    path_tracer::free();
    path_tracer::initialize(scene);
  }

  if (curr_iteration < render_state->total_iterations) {
    uchar4* pbo_dptr = NULL;
    curr_iteration++;
    cudaGLMapBufferObject(reinterpret_cast<void**>(&pbo_dptr), pbo);

    path_tracer::run(pbo_dptr, curr_iteration);

    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  } else {
    saveImage();
    path_tracer::free();
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
  }
}

void keyCallback(GLFWwindow* window,
                 int key,
                 int scancode,
                 int action,
                 int mods) {
  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camera_changed = true;
        render_state = &scene->state;
        Camera& cam = render_state->camera;
        cam.look_at = ogLookAt;
        break;
    }
  }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (MouseOverImGuiWindow()) {
    return;
  }

  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed =
      (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed =
      (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) {
    return;  // otherwise, clicking back into window causes re-start
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
    render_state = &scene->state;
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
    printf("Usage: %s SCENEFILE.json\n", argv[0]);
    return 1;
  }

  const char* sceneFile = argv[1];

  // Load scene file
  scene = new Scene(sceneFile);

  // Create Instance for ImGUIData
  gui_data = new GuiDataContainer();

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

  // Initialize CUDA and GL components
  init();

  // Initialize ImGui Data
  InitImguiData(gui_data);
  init_data_container(gui_data);

  run_main_loop();

  return EXIT_SUCCESS;
}
