/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

//#define M_PI 3.14159265359

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if(!is_initialized) {

      num_particles = 100;  // TODO: Set the number of particles
      // This line creates a normal (Gaussian) distribution for x
      normal_distribution<double> dist_x(x, std[0]);

      //Create normal distributions for y and theta
      normal_distribution<double> dist_y(y, std[1]);
      normal_distribution<double> dist_theta(theta, std[2]);

      std::default_random_engine gen;

      for (int i = 0; i < num_particles; ++i) {
        //Sample from these normal distributions like this: 
        double sample_x = dist_x(gen);
        double sample_y = dist_y(gen);
        double sample_theta = dist_theta(gen);
        //   where "gen" is the random engine initialized earlier.

        Particle p;
        p.id = i;
        p.x = sample_x;
        p.y = sample_y;
        p.theta = sample_theta;
        p.weight = 1.0;
        particles.push_back(p);
		weights.push_back(1.0);
      }
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  if(is_initialized){
    std::default_random_engine gen;


    for(int i=0;i<(int) particles.size();i++){
      //Add measurements to each particle.
      if(yaw_rate == 0) {
        particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      	particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      }
      else {

        particles[i].x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) -sin(particles[i].theta));
        particles[i].y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
        particles[i].theta = particles[i].theta + yaw_rate * delta_t;
      }
     // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    //Create normal distributions for y and theta
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]); 
      // add random Gaussian noise.
      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);

    }
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {

  for(int i=0;i<(int)observations.size();i++){
    double dist = 999999999.0;
    for(int j=0;j<(int)predicted.size();j++){

     // Find the predicted measurement that is closest to each observed measurement
     double tempdist = pow(observations[i].x - predicted[j].x, 2.0) + pow(observations[i].y - predicted[j].y, 2.0);
     if(tempdist < dist){
       dist = tempdist;
       //assign the observed measurement to this particular landmark.
       observations[i].id = j;
     }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   vector<LandmarkObs> observations, 
                                   const Map &map_landmarks) {
	//consider every particle as the estimated position of the car & caculate its weight 
  
	// Get landmarks within sensor range
  	// loop through all particles
	for(int i=0;i<num_particles;i++) {
      vector<LandmarkObs> landmarksWithinRange;
      
      // loop through all landmarks
      for(int j=0;j<(int)map_landmarks.landmark_list.size();j++){
        double dX = map_landmarks.landmark_list[j].x_f - particles[i].x;
        double dY = map_landmarks.landmark_list[j].y_f - particles[i].y;
        double dist = sqrt(dX * dX + dY * dY);
        
        if(dist <= sensor_range){
          LandmarkObs l;
          l.id = map_landmarks.landmark_list[j].id_i;
          l.x = dX * cos(particles[i].theta) + dY * sin(particles[i].theta);
          l.y = dY * cos(particles[i].theta) - dX * sin(particles[i].theta) ;
          landmarksWithinRange.push_back(l);
        }
      }
      //vector<LandmarkObs> observations
  //Associate every observation by its neigbouring landmarks
        dataAssociation(landmarksWithinRange, observations);
  
	
  double new_weight = 1.0;
  // For every observation check the matched landmark & calculate weight
  for(int j=0;j<(int)observations.size();j++){
	
    double dX = observations[j].x - landmarksWithinRange[observations[j].id].x;
    double dY = observations[j].y - landmarksWithinRange[observations[j].id].y;
    
    // calculate normalization term
    //Got help from
    //https://github.com/ByteShaker/CarND-Kidnapped-Vehicle-Project/blob/master/src/particle_filter.cpp
    double numerator = exp(- 0.5 * (pow(dX,2.0)*std_landmark[0] + pow(dY,2.0)*std_landmark[1] ));
    double denominator = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
    new_weight = new_weight * numerator/denominator;
    
    
  	}
  	weights[i] = new_weight; 
     particles[i].weight = new_weight;
	}
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Creating distributions.
  std::default_random_engine gen;

  // Using Resampling Wheel to resample particles
  vector <Particle> resampledParticles;
  for(int i=0;i<num_particles;i++) {
    std::discrete_distribution<> d(weights.begin(), weights.end());

    resampledParticles.push_back(particles[d(gen)]);
  }
  particles = resampledParticles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}