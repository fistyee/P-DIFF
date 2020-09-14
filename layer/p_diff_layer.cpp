#include <algorithm>
#include <cfloat>
#include <vector>
#include <assert.h>

#include "caffe/layers/p_diff_layer.hpp"

namespace caffe {
  template <class T>
  T p_diff_clamp(T x, T min, T max)
  {
      if (x > max) return max;
      if (x < min) return min;
      return x;
  }

  template <typename Dtype>
  Dtype PDIFFLayer<Dtype>::cal_delta(const Dtype* probs, const int dim, const int gt)
  {
      Dtype prob = probs[gt];
      //return prob;// m1
      Dtype nprob = 0;
      for (int i = 0; i < dim; i++)
      {
          if (i != gt && nprob < probs[i])
          {
              nprob = probs[i];
          }
      }
      Dtype delta = prob - nprob;
      return delta;// m2
  }

  template <typename Dtype>
  Dtype PDIFFLayer<Dtype>::delta2weight(const Dtype delta)
  {
      return get_bin_id(delta) >= t_bin_id_ ? 1.0 : 0.0;
  }

  template <typename Dtype>
  int PDIFFLayer<Dtype>::get_bin_id(const Dtype delta)
  {
      int bin_id = bins_ * (delta - value_low_) / (value_high_ - value_low_);
      bin_id = p_diff_clamp<int>(bin_id, 0, bins_);
      return bin_id;
  }

  template <typename Dtype>
  Dtype PDIFFLayer<Dtype>::get_delta(const int bin_id)
  {
      Dtype delta = value_low_ + (value_high_ - value_low_) * bin_id / bins_;
      delta = p_diff_clamp<Dtype>(delta, value_low_, value_high_);
      return delta;
  }

  template <typename Dtype>
  void PDIFFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                             const vector<Blob<Dtype>*>& top)
  {
      srand((unsigned)time(NULL));
      
      const PDIFFParameter& p_diff_param = this->layer_param_.p_diff_param();
      start_iter_ = p_diff_param.start_iter();// 1
      bins_ = p_diff_param.bins();// 2
      slide_batch_num_ = p_diff_param.slide_batch_num();// 3
      value_low_ = p_diff_param.value_low();// 4 delta_low
      value_high_ = p_diff_param.value_high();// 5 delta_high
      debug_ = p_diff_param.debug();// 6
      debug_prefix_ = p_diff_param.debug_prefix();// 7
      noise_ratio_ = p_diff_param.noise_ratio();// 8
      epoch_iters_ = p_diff_param.epoch_iters();// 9
      Tk_ = p_diff_param.t_k();// 10
      use_auto_noise_ratio_ = p_diff_param.use_auto_noise_ratio();// 11
      thred_train_sat_ = p_diff_param.thred_train_sat();// 12 zeta

      CHECK_GE(start_iter_, 1) << "start iteration must be large than or equal to 1";
      CHECK_GE(bins_, 1) << "bins must be large than or equal to 1";
      CHECK_GE(slide_batch_num_, 1) << "slide batch num must be large than or equal to 1";
      CHECK_GT(value_high_, value_low_) << "high value must be large than low value";
      
      iter_ = 0;
      T_ = 0;
      start_iter_ = std::max(start_iter_, slide_batch_num_);
      pdf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
      clean_pdf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
      noise_pdf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
      pcf_ = std::vector<Dtype>(bins_+1, Dtype(0.0));
  }

  template <typename Dtype>
  void PDIFFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                          const vector<Blob<Dtype>*>& top)
  {
      CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "The size of bottom[0] and bottom[1] don't match";
      
      if (top[0] != bottom[0]) top[0]->ReshapeLike(*bottom[0]);
      
      // weights
      vector<int> weights_shape(1);
      weights_shape[0] = bottom[0]->num();
      weights_.Reshape(weights_shape);
  }

  template <typename Dtype>
  void PDIFFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                              const vector<Blob<Dtype>*>& top)
  {
      const Dtype* noise_data = NULL;
      if (bottom.size() == 4)// just for show debugging information
      {
          noise_data = bottom[3]->cpu_data();
      }
      const Dtype* label_data = bottom[2]->cpu_data();
      const Dtype* prob_data = bottom[1]->cpu_data();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      Dtype* weight_data = weights_.mutable_cpu_data();

      int count = bottom[0]->count();
      int num = bottom[0]->num();// batch_size
      int dim = count / num;// c * h * w

      if (top[0] != bottom[0]) caffe_copy(count, bottom_data, top_data);
      
      if (this->phase_ != TRAIN) return;

      // update probability distribution/density function
      // add
      for (int i = 0; i < num; i++)
      {
          int gt = static_cast<int>(label_data[i]);
          int noise_gt = -1;
          if (noise_data)
          {
              noise_gt = static_cast<int>(noise_data[i]);
              queue_label_.push(noise_gt);
          }
          if (gt < 0)
          {
              queue_bin_id_.push(-1);
              continue;
          }

          int bin_id = get_bin_id(cal_delta(prob_data+i*dim, dim, gt));
          ++pdf_[bin_id];
          if (noise_data)
          {
              if (noise_gt == 0) ++clean_pdf_[bin_id];
              else ++noise_pdf_[bin_id];
          }
          queue_bin_id_.push(bin_id);
      }
      // del
      while (queue_bin_id_.size() > slide_batch_num_ * num)
      {
          int bin_id = queue_bin_id_.front();
          queue_bin_id_.pop();
          int noise_gt = -1;
          if (noise_data)
          {
              noise_gt = queue_label_.front();
              queue_label_.pop();
          }
          if (bin_id != -1)
          {
              --pdf_[bin_id];
              if (noise_data)
              {
                  if (noise_gt == 0) --clean_pdf_[bin_id];
                  else --noise_pdf_[bin_id];
              }
          }
      }

      // update probability cumulative function
      Dtype sum_pdf = 0.0;
      for (int i = 0; i <= bins_; i++)
      {
          sum_pdf += pdf_[i];
      }
      pcf_[0] = pdf_[0] / sum_pdf;
      for (int i = 1; i <= bins_; i++)
      {
          pcf_[i] = pcf_[i-1] + pdf_[i] / sum_pdf;
      }

      ++iter_;
      T_ = iter_ / epoch_iters_;
      if (iter_ < start_iter_) return;

      // compute noise ratio
      Dtype noise_ratio;
      if (use_auto_noise_ratio_)
      {
          Dtype train_sat = 0;
          for (int i = 0; i < bins_; i++)
          {
              Dtype p = (get_delta(i) + get_delta(i+1)) / 2.0;
              train_sat += fabs(p) * pdf_[i] / sum_pdf;
          }
          
          if (T_ <= Tk_ || train_sat < thred_train_sat_)
          {
              Dtype thred_prob = -1.0 + (0.0 + 1.0) * std::min(1.0, 1.0 * T_ / Tk_);
              noise_ratio_ = pcf_[get_bin_id(thred_prob)];
          }
          noise_ratio = noise_ratio_;

          if (debug_)
          {
              printf("debug: train_sat: %f, noise_ratio: %f\n", train_sat, noise_ratio);
          }
      }
      else
      {
          noise_ratio = noise_ratio_ * std::min(1.0, 1.0 * T_ / Tk_);
      }

      for (t_bin_id_ = 0; t_bin_id_ <= bins_ && pcf_[t_bin_id_] <= noise_ratio; ++t_bin_id_);

      // compute weights
      for (int i = 0; i < num; i++)
      {
          int gt = static_cast<int>(label_data[i]);
          if (gt < 0) continue;
          weight_data[i] = delta2weight(cal_delta(prob_data+i*dim, dim, gt));
      }

      // print debug information
      if (debug_ && iter_ % 100 == start_iter_ % 100)
      {
          /*
          char file_name[256];
          sprintf(file_name, "/tmp/log/%s_%d.txt", debug_prefix_.c_str(), iter_);
          FILE *log = fopen(file_name, "w");
          
          fprintf(log, "debug iterations: %d\n", iter_);
                printf("debug iterations: %d\n", iter_);
          
          fprintf(log, "noise_ratio: %.2f, threshould_delta: %.2f\n", noise_ratio, get_delta(t_bin_id_));
                printf("noise_ratio: %.2f, threshould_delta: %.2f\n", noise_ratio, get_delta(t_bin_id_));
          
          fprintf(log, "pdf:\n");
                printf("pdf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_delta(i), pdf_[i]);
                    printf("%.2f %.2f\n", get_delta(i), pdf_[i]);
          }
          
          fprintf(log, "clean pdf:\n");
                printf("clean pdf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_delta(i), clean_pdf_[i]);
                    printf("%.2f %.2f\n", get_delta(i), clean_pdf_[i]);
          }

          fprintf(log, "noise pdf:\n");
                printf("noise pdf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_delta(i), noise_pdf_[i]);
                    printf("%.2f %.2f\n", get_delta(i), noise_pdf_[i]);
          }

          fprintf(log, "pcf:\n");
                printf("pcf:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_delta(i), pcf_[i]);
                    printf("%.2f %.2f\n", get_delta(i), pcf_[i]);
          }
          
          fprintf(log, "weight:\n");
                printf("weight:\n");
          for (int i = 0; i <= bins_; i++)
          {
              fprintf(log, "%.2f %.2f\n", get_delta(i), delta2weight(get_delta(i)));
                    printf("%.2f %.2f\n", get_delta(i), delta2weight(get_delta(i)));
          }
          
          fclose(log);
          */
      }
  }

  template <typename Dtype>
  void PDIFFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down, 
                                               const vector<Blob<Dtype>*>& bottom)
  {
      if (propagate_down[0])
      {
          const Dtype* label_data = bottom[2]->cpu_data();
          const Dtype* top_diff = top[0]->cpu_diff();
          Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
          const Dtype* weight_data = weights_.cpu_data();

          int count = bottom[0]->count();
          int num = bottom[0]->num();
          int dim = count / num;

          if (top[0] != bottom[0]) caffe_copy(count, top_diff, bottom_diff);

          if (this->phase_ != TRAIN) return;

          if (iter_ < start_iter_) return;
          /*
          if (iter_ < start_iter_)
          {
              caffe_set(count, Dtype(0), bottom_diff);
              return;
          }
          */

          // backward
          for (int i = 0; i < num; i++)
          {
              int gt = static_cast<int>(label_data[i]);
              if (gt < 0) continue;
              for (int j = 0; j < dim; j++)
              {
                  bottom_diff[i * dim + j] *= weight_data[i];
              }
          }
      }
  }

#ifdef CPU_ONLY
  STUB_GPU(PDIFFLayer);
#endif

INSTANTIATE_CLASS(PDIFFLayer);
REGISTER_LAYER_CLASS(PDIFF);
}  // namespace caffe

