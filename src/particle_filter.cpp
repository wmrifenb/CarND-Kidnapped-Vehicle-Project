/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 50;

  normal_distribution<double> norm_dist_x(x, std[0]);
  normal_distribution<double> norm_dist_y(y, std[1]);
  normal_distribution<double> norm_dist_theta(theta, std[2]);
  default_random_engine gen;

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = norm_dist_x(gen);
    particle.y = norm_dist_y(gen);
    particle.theta = norm_dist_theta(gen);
    particle.weight = 1;
    particles.push_back(particle);
    weights.push_back(1);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> norm_dist_x(0, std_pos[0]);
  normal_distribution<double> norm_dist_y(0, std_pos[1]);
  normal_distribution<double> norm_dist_theta(0, std_pos[2]);
  default_random_engine gen;

  //Prevent divide by zero
  if (fabs(yaw_rate) < 0.0001) {
    yaw_rate = 0.0001;
  }

  for (int i = 0; i < num_particles; i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    //Use motion model for prediction
    x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
    y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
    theta += yaw_rate * delta_t;

    //Add gaussian noise
    particles[i].x = x + norm_dist_x(gen);
    particles[i].y = y + norm_dist_y(gen);
    particles[i].theta = theta + norm_dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  //Assign nearest neighbor prediction to observations
  for (int i = 0; i < observations.size(); i++) {
    double min_distance = 1E9;
    for (int j = 0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y,
                             predicted[j].x, predicted[j].y);
      if (distance < min_distance) {
        min_distance = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];

    //Convert observations to map frame
    vector<LandmarkObs> observations_map_frame;

    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs obs = observations[j];
      LandmarkObs obs_map_frame;
      obs_map_frame.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
      obs_map_frame.y = obs.y * cos(p.theta) + obs.x * sin(p.theta) + p.y;
      obs_map_frame.id = 0;
      observations_map_frame.push_back(obs_map_frame);
    }

    //Find all landmarks within sensor range
    vector<LandmarkObs> predicted;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s l = map_landmarks.landmark_list[j];
      double distance = dist(p.x, p.y, l.x_f, l.y_f);
      if (distance < sensor_range) {
        LandmarkObs obs;
        obs.x = l.x_f;
        obs.y = l.y_f;
        obs.id = l.id_i;
        predicted.push_back(obs);
      }
    }

    if (predicted.size() > 0) {
      //Associate each observation to nearest neighbor prediction
      dataAssociation(predicted, observations_map_frame);

      //Use product of multivariate gaussion probability density of each observation to assign weight to each particle
      p.weight = 1;

      for (int j = 0; j < observations_map_frame.size(); j++) {
        LandmarkObs obs = observations_map_frame[j];
        Map::single_landmark_s l = map_landmarks.landmark_list[obs.id - 1];
        double weight = exp(
            -((pow(l.x_f - obs.x, 2) / (2 * pow(std_landmark[0], 2))
                + pow(l.y_f - obs.y, 2) / (2 * pow(std_landmark[1], 2)))))
            / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        p.weight *= weight;
      }
    } else {
      p.weight = 0;
    }

    weights[i] = p.weight;
    particles[i].weight = p.weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> resampled_particles;
  default_random_engine gen;
  discrete_distribution<> distribution(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++) {
    int number = distribution(gen);
    resampled_particles.push_back(particles[number]);
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
