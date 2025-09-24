CUDA Path Tracer
================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Charles Wang
  * [LinkedIn](https://linkedin.com/in/zwcharl)
  * [Personal website](https://charleszw.com)
* Tested on:
  * Windows 11 Pro (26100.4946)
  * Ryzen 5 7600X @ 4.7Ghz
  * 32 GB RAM
  * RTX 5060 Ti 16 GB (Studio Driver 581.29)

# CUDA Path Tracer

## Program structure

- Path tracer is run for a certain number of iterations. This is determined by the scene file we're currently running
- Each iteration, represented by one call to `path_tracer::run`, first generates the initial set of rays coming out of the camera
- Then we enter a loop that does the following:
  - Using the calculated rays, compute potential intersections with the scene
  - Stream compaction away dead paths, i.e. paths that did not intersect with anything or went out of bounds
  - Add the light and color contribution to the path
  - Determine the next ray to travel to
- We exit the loop if we've either hit the max accepted depth (this should be kinda rare) or we've stream compacted away all paths (more likely to occur)
- Gather pixel color from each path by appending it to all previous contributions and display it

## Features

Roughly organized in chronological order of when I first implemented it.

### Cosine-weighted hemisphere sampling and Lambertian diffuse materials

After playing with the base code a bit and getting a sense of the overall project structure, I implemented my first material, a very simple Lambertian diffuse shading model.

- probably some explanation of what it is
- BSDF is constant with respect to the incident angle, and is simply $\frac{\text{albedo}}{\pi}$

A function for cosine-weighted hemisphere sampling was already provided for us, so I didn't bother trying uniform hemisphere sampling first. The PDF for cosine-weighted sampling is given by $\frac{\text{abs}(\cos{\theta})}{\pi}$, which we also have to consider in the lighting equation.

- probably explain why cosine-weighted hemisphere sampling is better than uniform sampling

Then, the overall throughput contribution for this intersection is given by `bsdf * lambert / pdf`, where `lambert` is simply $\text{abs}(\cos{\theta})$.[^1] With no other changes, testing on the default `cornell.json` scene gives us this:

![](renders/lambertian_cosine_weighted/cornell.2025-09-23_00-38-16z.5000samp.png)

5000 iterations.

### Discarding out of bounds intersections

TODO(aczw): use difference as metric to measure how effective sampling methods are?

## Building

I've somewhat modified the [CMakeLists.txt](CMakeLists.txt) file. Here are the changes that I've made:

- Moved the `include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")` call out of the `if(UNIX)` branch to make it available on Windows as well
- Renamed and moved various file and updated `headers` and `sources` accordingly.

[^1]: This `lambert` term should not to be confused with the Lambertian diffuse model. It's part of the overall light transport equation and must be computed for all materials.
