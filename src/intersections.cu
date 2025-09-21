#include "intersections.h"

__host__ __device__ float box_intersection_test(Geom box,
                                                Ray r,
                                                glm::vec3& intersection_point,
                                                glm::vec3& normal,
                                                bool& outside) {
  Ray q;
  q.origin = multiply_mat4_vec4(box.inverseTransform, glm::vec4(r.origin, 1.0f));
  q.direction = glm::normalize(multiply_mat4_vec4(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

  float t_min = -1e38f;
  float t_max = 1e38f;
  glm::vec3 t_min_n;
  glm::vec3 t_max_n;

  for (int xyz = 0; xyz < 3; ++xyz) {
    float qdxyz = q.direction[xyz];
    /*if (glm::abs(qdxyz) > 0.00001f)*/
    {
      float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
      float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
      float ta = glm::min(t1, t2);
      float tb = glm::max(t1, t2);
      glm::vec3 n;
      n[xyz] = t2 < t1 ? +1 : -1;
      if (ta > 0 && ta > t_min) {
        t_min = ta;
        t_min_n = n;
      }
      if (tb < t_max) {
        t_max = tb;
        t_max_n = n;
      }
    }
  }

  if (t_max >= t_min && t_max > 0) {
    outside = true;
    if (t_min <= 0) {
      t_min = t_max;
      t_min_n = t_max_n;
      outside = false;
    }
    intersection_point = multiply_mat4_vec4(box.transform, glm::vec4(get_point_on_ray(q, t_min), 1.0f));
    normal = glm::normalize(multiply_mat4_vec4(box.invTranspose, glm::vec4(t_min_n, 0.0f)));
    return glm::length(r.origin - intersection_point);
  }

  return -1;
}

__host__ __device__ float sphere_intersection_test(Geom sphere,
                                                   Ray r,
                                                   glm::vec3& intersection_point,
                                                   glm::vec3& normal,
                                                   bool& outside) {
  float radius = .5;

  glm::vec3 ro = multiply_mat4_vec4(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiply_mat4_vec4(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

  Ray rt;
  rt.origin = ro;
  rt.direction = rd;

  float vector_dot_direction = glm::dot(rt.origin, rt.direction);
  float radicand = vector_dot_direction * vector_dot_direction - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
  if (radicand < 0) {
    return -1;
  }

  float square_root = sqrt(radicand);
  float first_term = -vector_dot_direction;
  float t1 = first_term + square_root;
  float t2 = first_term - square_root;

  float t = 0;
  if (t1 < 0 && t2 < 0) {
    return -1;
  } else if (t1 > 0 && t2 > 0) {
    t = min(t1, t2);
    outside = true;
  } else {
    t = max(t1, t2);
    outside = false;
  }

  glm::vec3 obj_space_intersection = get_point_on_ray(rt, t);

  intersection_point = multiply_mat4_vec4(sphere.transform, glm::vec4(obj_space_intersection, 1.f));
  normal = glm::normalize(multiply_mat4_vec4(sphere.invTranspose, glm::vec4(obj_space_intersection, 0.f)));
  if (!outside) {
    normal = -normal;
  }

  return glm::length(r.origin - intersection_point);
}
