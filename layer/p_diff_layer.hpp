#ifndef CAFFE_P_DIFF_LAYER_HPP_
#define CAFFE_P_DIFF_LAYER_HPP_

#include <vector>
#include <queue>
#include <algorithm>
#include <cstdlib>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace caffe {

  template <typename Dtype>
  class PDIFFLayer : public Layer<Dtype> {
  public:
    explicit PDIFFLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "PDIFF"; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    //virtual inline int ExactNumBottomBlobs() const { return 3; }
    virtual inline int MinBottomBlobs() const { return 3; }
    virtual inline int MaxBottomBlobs() const { return 4; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //                         const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Dtype cal_delta(const Dtype* probs, const int dim, const int gt);

    Dtype delta2weight(const Dtype delta);
    int get_bin_id(const Dtype delta);
    Dtype get_delta(const int bin_id);

    Dtype mean_filter(const int bin_id);

    int start_iter_;// 1
    int bins_;// 2
    int slide_batch_num_;// 3
    Dtype value_low_;// 4
    Dtype value_high_;// 5
    bool debug_;// 6
    std::string debug_prefix_;// 7
    Dtype noise_ratio_;// 8
    int epoch_iters_;// 9
    int Tk_;// 10
    bool use_auto_noise_ratio_;// 11
    Dtype thred_train_sat_;// 12

    int iter_;
    int T_;
    int t_bin_id_;
    std::queue<int> queue_label_;
    std::queue<int> queue_bin_id_;
    std::vector<Dtype> pdf_;
    std::vector<Dtype> clean_pdf_;
    std::vector<Dtype> noise_pdf_;
    std::vector<Dtype> pcf_;
    Blob<Dtype> weights_;
  };

}  // namespace caffe

#endif  // CAFFE_P_DIFF_LAYER_HPP_
