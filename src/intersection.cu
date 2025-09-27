#include "intersection.cuh"

#include <cuda/std/limits>

__host__ __device__ cuda::std::optional<Hit> test_cube_hit(Geometry cube, Ray ray) {
  float t_min = cuda::std::numeric_limits<float>::lowest();
  float t_max = cuda::std::numeric_limits<float>::max();
  glm::vec3 t_min_n;
  glm::vec3 t_max_n;

  Ray q = {
      .origin = glm::vec3(cube.inv_transform * glm::vec4(ray.origin, 1.f)),
      .direction = glm::normalize(glm::vec3(cube.inv_transform * glm::vec4(ray.direction, 0.f))),
  };

  for (int xyz = 0; xyz < 3; ++xyz) {
    float qdxyz = q.direction[xyz];

    /*if (glm::abs(qdxyz) > 0.00001f)*/
    {
      float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
      float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
      float ta = glm::min(t1, t2);
      float tb = glm::max(t1, t2);
      glm::vec3 n;
      n[xyz] = t2 < t1 ? 1 : -1;
      if (ta > 0.f && ta > t_min) {
        t_min = ta;
        t_min_n = n;
      }
      if (tb < t_max) {
        t_max = tb;
        t_max_n = n;
      }
    }
  }

  if (t_max >= t_min && t_max > 0.f) {
    Hit hit;
    hit.surface = Surface::Outside;

    if (t_min <= 0.f) {
      t_min = t_max;
      t_min_n = t_max_n;
      hit.surface = Surface::Inside;
    }

    hit.point = glm::vec3(cube.transform * glm::vec4(q.get_point(t_min), 1.f));
    hit.normal = glm::normalize(glm::vec3(cube.inv_transpose * glm::vec4(t_min_n, 0.f)));
    hit.t = glm::length(ray.origin - hit.point);
    hit.material_id = cube.material_id;

    return hit;
  }

  return {};
}

__host__ __device__ cuda::std::optional<Hit> test_sphere_hit(Geometry sphere, Ray ray) {
  static const float radius = 0.5f;

  Ray rt = {
      .origin = glm::vec3(sphere.inv_transform * glm::vec4(ray.origin, 1.f)),
      .direction = glm::normalize(glm::vec3(sphere.inv_transform * glm::vec4(ray.direction, 0.f))),
  };

  float vector_dot_direction = glm::dot(rt.origin, rt.direction);
  float radicand = vector_dot_direction * vector_dot_direction -
                   (glm::dot(rt.origin, rt.origin) - std::powf(radius, 2));

  if (radicand < 0.f) {
    return {};
  }

  float square_root = std::sqrt(radicand);
  float first_term = -vector_dot_direction;
  float t1 = first_term + square_root;
  float t2 = first_term - square_root;

  Hit hit;

  float t = 0.f;
  if (t1 < 0.f && t2 < 0.f) {
    return {};
  } else if (t1 > 0.f && t2 > 0.f) {
    t = glm::min(t1, t2);
    hit.surface = Surface::Outside;
  } else {
    t = glm::max(t1, t2);
    hit.surface = Surface::Inside;
  }

  glm::vec3 obj_space_point = rt.get_point(t);

  hit.point = glm::vec3(sphere.transform * glm::vec4(obj_space_point, 1.f));
  hit.t = glm::length(ray.origin - hit.point);
  hit.normal = glm::normalize(glm::vec3(sphere.inv_transpose * glm::vec4(obj_space_point, 0.f)));
  hit.material_id = sphere.material_id;

  // TODO(aczw): is this necessary...
  // if (hit.surface == Surface::Inside) {
  //   hit.normal = -hit.normal;
  // }

  return hit;
}
