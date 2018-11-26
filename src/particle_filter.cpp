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
	default_random_engine gen;
  	normal_distribution<double> dist_x(x, std[0]);
  	normal_distribution<double> dist_y(y, std[1]);
  	normal_distribution<double> dist_theta(theta, std[2]);
	num_particles = 250;
  	if(is_initialized == true)
    {
      return;
    }
  	else
    {
     	for(unsigned int i = 1; i <= num_particles; i++)
        {
         	Particle particle;  
         	particle.id = i;
          	particle.x = dist_x(gen);
          	particle.y = dist_y(gen);
          	particle.theta = dist_theta(gen);
          	particle.weight = 1;
          	particles.push_back(particle);
        }
      	is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
  	normal_distribution<double> dist_x(0, std_pos[0]);
  	normal_distribution<double> dist_y(0, std_pos[1]);
  	normal_distribution<double> dist_theta(0, std_pos[2]);
  	for(unsigned int i = 0; i < particles.size(); i++)
    {

     if(fabs(yaw_rate) < 0.0001)
     {
		particles[i].x = particles[i].x + ((velocity*delta_t)*(cos(particles[i].theta)) + dist_x(gen));
		particles[i].y = particles[i].y + ((velocity*delta_t)*(sin(particles[i].theta)) + dist_y(gen));     
		particles[i].theta = particles[i].theta + dist_theta(gen);
     }
     else
     {
		particles[i].x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta)) +
          dist_x(gen);
       	particles[i].y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t))) +
          dist_y(gen);
       	particles[i].theta = particles[i].theta + (yaw_rate*delta_t) + dist_theta(gen);
     }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  	 
  	for(unsigned int i = 0; i < observations.size(); i++)
    {
      double min_dist = sqrt(pow((observations[i].x - predicted[0].x), 2) + pow((observations[i].y - predicted[0].y), 2)); 
      observations[i].id = predicted[0].id;
      for(unsigned int k = 1; k < predicted.size(); k++)
      {
		double dist = sqrt(pow((observations[i].x - predicted[k].x), 2) + pow((observations[i].y - predicted[k].y), 2)); 
        if(dist < min_dist)
        {
          min_dist = dist; 
          observations[i].id = predicted[k].id;
        }
      }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

  	for(unsigned int i = 0; i < particles.size(); i++)
    {
      particles[i].weight = 1;
      std::vector<LandmarkObs> filtered_landmarks;
      std::vector<LandmarkObs> map_sys_observations = observations; 
      for(unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++)
      {
        
        double distance = sqrt(pow(particles[i].x - map_landmarks.landmark_list[k].x_f,2)
                               + pow(particles[i].y - map_landmarks.landmark_list[k].y_f,2));
        if(distance <= sensor_range)
        {
          filtered_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[k].id_i, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f});
        }
      }
      for(unsigned int k = 0; k < observations.size(); k ++)
      {
        map_sys_observations[k].x  = particles[i].x +
          (cos(particles[i].theta)*observations[k].x) - (sin(particles[i].theta) * observations[k].y);
        map_sys_observations[k].y  = particles[i].y +
          (sin(particles[i].theta)*observations[k].x) + (cos(particles[i].theta) * observations[k].y);           
      }
      dataAssociation(filtered_landmarks, map_sys_observations);
      
      for(unsigned int k = 0; k < map_sys_observations.size(); k++)
      {
        for(unsigned int j = 0; j < filtered_landmarks.size(); j++)
        {
          if(map_sys_observations[k].id == filtered_landmarks[j].id)
          {
            double weight = 1/(2*std_landmark[0]*std_landmark[1])*
              exp(-(pow(map_sys_observations[k].x - filtered_landmarks[j].x,2)/
                    pow(2*std_landmark[0], 2) + pow( map_sys_observations[k].y - filtered_landmarks[j].y,2)/
                    pow(2*std_landmark[1],2)));
            if(weight == 0)
            {
              particles[i].weight = particles[i].weight * 0.0001;
            }
            else
            {
              particles[i].weight = particles[i].weight * weight;
            }
          	break;
          }
        }
      }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	double max_weight = 0;
	std::uniform_int_distribution<int> dist_index(0, num_particles-1);
	double beta = 0;
	int index = dist_index(gen);
  	vector<Particle> new_particles;
  	for(unsigned int i = 0; i < particles.size(); i++)
    {
      if(particles[i].weight > max_weight)
      {
        max_weight = particles[i].weight;
      }
    }
	std::uniform_real_distribution<double> dist_weight(0, max_weight);

	for(unsigned int i = 0; i < particles.size(); i++)
    {
      beta += dist_weight(gen)*2;
     while(particles[index].weight < beta)
     {
        beta -= particles[index].weight;
        index = (index + 1) % num_particles;
     }
      new_particles.push_back(particles[index]);
    }
	particles = new_particles;  
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
  	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
