// Implements the Hungarian algorithm.
// Batch usage:
// Input is a batch of weight matrice, W: [B, N_X, N_Y]
// Output is a matching, M: [B, N_X, N_Y], and  vertex covers C_X: [B, N_X, 1],
// C_Y: [B, 1, N_Y]
//
// Single example usage:
// Input is a 2-D weight matrix W: [N_X, N_Y].
// Output is a matching M: [N_X, N_Y], and vertex covers C_X: [B, N_X, 1], C_Y:
// [B, 1, N_Y].

#include <deque>
#include <iostream>
#include <limits>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#define EPSILON 1e-6
#define ABS(x) (((x) > 0) ? (x) : -(x))
#define MAX_NUM_ITERATION 1000

using namespace tensorflow;

typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> MatrixXfR;

REGISTER_OP("Hungarian")
    .Input("weights: float")
    .Output("matching: float")
    .Output("cover_x: float")
    .Output("cover_y: float");

class HungarianOp : public OpKernel {
 public:
  explicit HungarianOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const auto& shape = input_tensor.shape();

    // Create an output tensor
    Tensor* matching_tensor = NULL;
    Tensor* cover_x_tensor = NULL;
    Tensor* cover_y_tensor = NULL;

    TensorShape shape_x;
    TensorShape shape_y;
    int num_dim = shape.dims();
    int n_x;
    int n_y;

    if (num_dim == 3) {
      int num_ex = shape.dim_size(0);
      n_x = shape.dim_size(1);
      n_y = shape.dim_size(2);
      shape_x.AddDim(num_ex);
      shape_y.AddDim(num_ex);
    } else if (num_dim == 2) {
      n_x = shape.dim_size(0);
      n_y = shape.dim_size(1);
    } else {
      LOG(FATAL) << "Must have dimension 3 or 2.";
      return;
    }

    shape_x.AddDim(n_x);
    shape_x.AddDim(1);
    shape_y.AddDim(1);
    shape_y.AddDim(n_y);

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape, &matching_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, shape_x, &cover_x_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, shape_y, &cover_y_tensor));

    if (num_dim == 3) {
      ComputeHungarianBatch(input_tensor, matching_tensor, cover_x_tensor,
                            cover_y_tensor);
    } else if (num_dim == 2) {
      ComputeHungarian(input_tensor, matching_tensor, cover_x_tensor,
                       cover_y_tensor);
    }
  }

 private:
  MatrixXfR CopyInput(const Tensor& tensor) {
    const auto& shape = tensor.shape();
    MatrixXfR copy =
        Eigen::Map<MatrixXfR>((float*)tensor.tensor_data().data(),
                              shape.dim_size(0), shape.dim_size(1));

    return copy;
  }

  void CopyOutput(const MatrixXfR& output, Tensor* output_tensor) {
    auto output_matrix = output_tensor->matrix<float>();
    const auto& shape = output_tensor->shape();
    for (int i = 0; i < shape.dim_size(0); ++i) {
      for (int j = 0; j < shape.dim_size(1); ++j) {
        output_matrix(i, j) = output(i, j);
      }
    }
  }

  bool Augment(const MatrixXfR& capacity, MatrixXfR& flow,
               MatrixXfR& residual) {
    int n = residual.outerSize();
    int s = 0;
    int t = n - 1;

    std::deque<int> q;
    q.push_back(s);

    bool* mark = (bool*)calloc(n, sizeof(bool));
    int* p = (int*)calloc(n, sizeof(int));
    bool found = false;

    for (int v = 0; v < n; ++v) {
      p[v] = -1;
    }

    for (int i = 0; q.size() > 0 && i <= MAX_NUM_ITERATION; ++i) {
      if (i == MAX_NUM_ITERATION) {
        LOG(FATAL) << "Max number of iteration reached at BFS.";
      }
      int v = q.front();
      q.pop_front();
      mark[v] = true;
      if (v == t) {
        found = true;
        break;
      }
      for (int u = 0; u < n; ++u) {
        if (!mark[u] && residual(v, u) > 0) {
          q.push_back(u);
          p[u] = v;
        }
      }
    }

    if (found) {
      float b = capacity.maxCoeff();
      int v = t;
      for (int i = 0; p[v] != -1 && i <= MAX_NUM_ITERATION; ++i) {
        if (i == MAX_NUM_ITERATION) {
          LOG(FATAL)
              << "Max number of iteration reached at search parent list.";
        }
        b = MIN(b, residual(p[v], v));
        v = p[v];
      }

      v = t;
      for (int i = 0; p[v] != -1 && i <= MAX_NUM_ITERATION; ++i) {
        if (i == MAX_NUM_ITERATION) {
          LOG(FATAL)
              << "Max number of iteration reached at search parent list.";
        }
        if (capacity(p[v], v) > 0) {
          flow(p[v], v) += b;
        } else {
          flow(v, p[v]) -= b;
        }
        residual(p[v], v) -= b;
        residual(v, p[v]) += b;
        v = p[v];
      }
    }

    delete mark;
    delete p;
    VLOG(2) << "Found augmenting path";

    return found;
  }

  MatrixXfR MaxFlow(const MatrixXfR& capacity) {
    int n = capacity.outerSize();
    MatrixXfR flow = MatrixXfR::Zero(n, n);
    MatrixXfR residual(capacity);

    for (int i = 0; Augment(capacity, flow, residual) && i <= MAX_NUM_ITERATION;
         ++i) {
      if (i == MAX_NUM_ITERATION) {
        LOG(FATAL) << "Max number of iteration reached at max flow.";
      }
    }

    return flow;
  }

  void MaxBipartiteMatching(const MatrixXfR& graph, MatrixXfR* matching) {
    int n_X = graph.outerSize();
    int n_Y = graph.innerSize();
    int n = n_X + n_Y + 2;
    MatrixXfR capacity = MatrixXfR::Zero(n, n);
    int s = 0;
    int t = n_X + n_Y + 1;
    int x_start = 1;
    int y_start = n_X + 1;
    MatrixXfR ones = MatrixXfR::Constant(n, n, 1.0);
    capacity.block(x_start, y_start, n_X, n_Y) = graph.block(0, 0, n_X, n_Y);
    capacity.block(s, x_start, 1, n_X) = ones.block(s, x_start, 1, n_X);
    capacity.block(y_start, t, n_Y, 1) = ones.block(y_start, t, n_Y, 1);
    VLOG(2) << "reformed graph: \n" << capacity;

    MatrixXfR flow_max = MaxFlow(capacity);
    VLOG(2) << "max flow: \n" << flow_max;

    // MatrixXfR matching = MatrixXfR::Zero(n_X, n_Y);
    matching->block(0, 0, n_X, n_Y) =
        flow_max.block(x_start, y_start, n_X, n_Y);
    VLOG(2) << "matching: \n" << *matching;
    VLOG(2) << "saturate: " << IsBipartiteMatchingSaturate(*matching);
  }

  bool IsBipartiteMatchingSaturate(const MatrixXfR& matching) {
    int n_X = matching.outerSize();
    int n_Y = matching.innerSize();

    if (n_X >= n_Y) {
      // Each vertex in Y needs to match to vertex in X.
      for (int j = 0; j < n_Y; ++j) {
        float sum = 0;
        for (int i = 0; i < n_X; ++i) {
          sum += matching(i, j);
        }
        if (sum == 0) {
          return false;
        }
      }
      return true;
    } else {
      // Each vertex in X needs to match to vertex in Y.
      for (int i = 0; i < n_X; ++i) {
        float sum = 0;
        for (int j = 0; j < n_Y; ++j) {
          sum += matching(i, j);
        }
        if (sum == 0) {
          return false;
        }
      }
      return true;
    }
  }

  void GetSetBipartiteNeighbours(const std::set<int>& set,
                                 const MatrixXfR& graph,
                                 std::set<int>* neighbours) {
    neighbours->clear();
    int n_Y = graph.innerSize();
    for (auto it = set.begin(); it != set.end(); ++it) {
      int v = *it;
      for (int u = 0; u < n_Y; ++u) {
        if (graph(v, u) > 0) {
          neighbours->insert(u);
        }
      }
    }
  }

  bool SetContains(const std::set<int>& s, int elem) {
    return !(s.find(elem) == s.end());
  }

  bool SetEquals(const std::set<int>& a, const std::set<int>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (auto it = a.begin(); it != a.end(); ++it) {
      if (!SetContains(b, *it)) {
        return false;
      }
    }
    return true;
  }

  void PrintSet(const std::set<int>& s) {
    std::cout << "{";
    for (auto it = s.begin(); it != s.end(); ++it) {
      std::cout << *it << ", ";
    }
    std::cout << "}" << std::endl;
  }

  int GetMatchedX(int y, const MatrixXfR& matching) {
    int n_X = matching.outerSize();
    for (int u = 0; u < n_X; ++u) {
      if (matching(u, y) == 1.0) {
        return u;
      }
    }
    return -1;
  }

  int GetMatchedY(int x, const MatrixXfR& matching) {
    int n_Y = matching.innerSize();
    for (int v = 0; v < n_Y; ++v) {
      if (matching(x, v) == 1.0) {
        return v;
      }
    }
    return -1;
  }

  MatrixXfR GetEqualityGraph(const MatrixXfR& weights, const MatrixXfR& cover_x,
                             const MatrixXfR& cover_y) {
    int n_X = weights.outerSize();
    int n_Y = weights.innerSize();
    MatrixXfR equality = MatrixXfR::Zero(n_X, n_Y);
    for (int x = 0; x < n_X; ++x) {
      for (int y = 0; y < n_Y; ++y) {
        VLOG(2) << "x: " << x << " y: " << y << " cx: " << cover_x(x, 0)
                << " cy: " << cover_y(0, y) << " w: " << weights(x, y);
        if (ABS(cover_x(x, 0) + cover_y(0, y) - weights(x, y)) <= EPSILON &&
            (cover_x(x, 0) > 0 || cover_y(0, y) > 0)) {
          equality(x, y) = 1.0;
        }
      }
    }
    return equality;
  }

  void CopyMatrix(const MatrixXfR& src, MatrixXfR* dst) {
    for (int i = 0; i < src.outerSize(); ++i) {
      for (int j = 0; j < src.innerSize(); ++j) {
        (*dst)(i, j) = src(i, j);
      }
    }
  }

  void MinWeightedBipartiteCover(const MatrixXfR& weights, MatrixXfR* matching,
                                 MatrixXfR* cover_x, MatrixXfR* cover_y) {
    int n_X = weights.outerSize();
    int n_Y = weights.innerSize();
    MatrixXfR maxCoeff = weights.rowwise().maxCoeff();
    MatrixXfR& c_x = *cover_x;
    MatrixXfR& c_y = *cover_y;
    MatrixXfR& M = *matching;
    for (int x = 0; x < n_X; ++x) {
      c_x(x, 0) = maxCoeff(x, 0);
    }
    for (int y = 0; y < n_Y; ++y) {
      c_y(0, y) = 0.0f;
    }
    for (int x = 0; x < n_X; ++x) {
      for (int y = 0; y < n_Y; ++y) {
        M(x, y) = 0.0f;
      }
    }
    VLOG(1) << "initial cover x: \n" << c_x;
    VLOG(1) << "initial cover y: \n" << c_y;

    MatrixXfR equality(n_X, n_Y);
    std::set<int> S;
    std::set<int> T;
    bool next_match = true;

    for (int i = 0; i <= MAX_NUM_ITERATION; ++i) {
      if (i == MAX_NUM_ITERATION) {
        LOG(ERROR) << "Max number of iteration reached. Exit iteration "
                      "possibly due to non-termination condition.";
        LOG(ERROR) << "Input: " << weights;
        LOG(ERROR) << "Matching: " << *matching;
        LOG(ERROR) << "Equality: " << equality;
        LOG(ERROR) << "S: ";
        PrintSet(S);
        LOG(ERROR) << "T: ";
        PrintSet(T);
        LOG(ERROR) << "Exit";
        // Just return the unfinished matching here.
        // Other loops will be fatal, this one willl not.
        break;
      }
      VLOG(1) << "-----------------------------";
      VLOG(1) << "iteration " << i;
      VLOG(1) << "input graph: \n" << weights;
      VLOG(1) << "cover x: \n" << c_x;
      VLOG(1) << "cover y: \n" << c_y;
      equality = GetEqualityGraph(weights, c_x, c_y);
      VLOG(1) << "equality graph: \n" << equality;
      if (next_match) {
        MaxBipartiteMatching(equality, matching);

        if (IsBipartiteMatchingSaturate(M)) {
          VLOG(1) << "found solution, exit";
          VLOG(1) << "-----------------------------";
          return;
        }

        for (int u = 0; u < n_X; ++u) {
          if (GetMatchedY(u, M) == -1) {
            S.clear();
            S.insert(u);
            VLOG(1) << "Clearing S and T";
            VLOG(1) << "Adding " << u << " into S";
            T.clear();
            break;
          }
        }
      }

      std::set<int> N_S;
      GetSetBipartiteNeighbours(S, equality, &N_S);
      VLOG(1) << "S: ";
      // PrintSet(S);
      VLOG(1) << "T: ";
      // PrintSet(T);
      VLOG(1) << "N_S: ";
      // PrintSet(N_S);

      if (SetEquals(N_S, T)) {
        VLOG(1) << "N_S == T";
        VLOG(1) << "Update cover";
        float a = std::numeric_limits<float>::max();
        for (auto it = S.begin(); it != S.end(); ++it) {
          int x = *it;
          for (int y = 0; y < n_Y; ++y) {
            if (!SetContains(T, y)) {
              a = MIN(a, c_x(x, 0) + c_y(0, y) - weights(x, y));
            }
          }
        }
        VLOG(1) << "a: " << a;
        if (a < EPSILON) {
          next_match = true;
          continue;
        }
        for (auto it = S.begin(); it != S.end(); ++it) {
          int x = *it;
          VLOG(1) << "Update X cover " << x;
          c_x(x, 0) -= a;
        }
        for (auto it = T.begin(); it != T.end(); ++it) {
          int y = *it;
          VLOG(1) << "Update Y cover " << y;
          c_y(0, y) += a;
        }
        VLOG(1) << "cover x: \n" << c_x;
        VLOG(1) << "cover y: \n" << c_y;
      } else {
        VLOG(1) << "N_S != T";
        for (int j = 0; N_S.size() > T.size() && j <= MAX_NUM_ITERATION; ++j) {
          if (j == MAX_NUM_ITERATION) {
            LOG(FATAL)
                << "Max number of iteration reached at equalizing N_S, T.";
          }
          int y;
          for (auto it = N_S.begin(); it != N_S.end(); ++it) {
            y = *it;
            if (!SetContains(T, y)) {
              VLOG(1) << "pick y in N_S not in T: " << y;
              break;
            }
          }

          int z = GetMatchedX(y, M);
          if (z == -1) {
            VLOG(1) << "y unmatched, look for matching";
            next_match = true;
            break;
          } else {
            VLOG(1) << "y matched, increase S and T";
            next_match = false;
            S.insert(z);
            for (int v = 0; v < n_Y; ++v) {
              if (equality(z, v) > 0.0) {
                N_S.insert(v);
              }
            }
            T.insert(y);
            VLOG(1) << "S: ";
            // PrintSet(S);
            VLOG(1) << "T: ";
            // PrintSet(T);
            VLOG(1) << "N_S: ";
            // PrintSet(N_S);
          }
        }
      }

      VLOG(1) << "end of iteration";
      VLOG(1) << "-----------------------------";
    }
  }

  void ComputeHungarian(const Tensor& input_tensor, Tensor* matching_tensor,
                        Tensor* cover_x_tensor, Tensor* cover_y_tensor) {
    const auto& inp = CopyInput(input_tensor);
    const auto& shape = input_tensor.shape();
    int n_x = shape.dim_size(0);
    int n_y = shape.dim_size(1);
    MatrixXfR cover_x = MatrixXfR::Zero(n_x, 1);
    MatrixXfR cover_y = MatrixXfR::Zero(1, n_y);
    MatrixXfR matching = MatrixXfR::Zero(n_x, n_y);

    MinWeightedBipartiteCover(inp, &matching, &cover_x, &cover_y);
    CopyOutput(matching, matching_tensor);
    CopyOutput(cover_x, cover_x_tensor);
    CopyOutput(cover_y, cover_y_tensor);
  }

  void ComputeHungarianBatch(const Tensor& input_tensor,
                             Tensor* matching_tensor, Tensor* cover_x_tensor,
                             Tensor* cover_y_tensor) {
    const auto& shape = input_tensor.shape();
    int num_ex = shape.dim_size(0);
    int n_x = shape.dim_size(1);
    int n_y = shape.dim_size(2);
    const auto& inp = input_tensor.tensor<float, 3>();
    auto matching = matching_tensor->tensor<float, 3>();
    auto cover_x = cover_x_tensor->tensor<float, 3>();
    auto cover_y = cover_y_tensor->tensor<float, 3>();

    for (int i = 0; i < num_ex; ++i) {
      MatrixXfR c_x = MatrixXfR::Zero(n_x, 1);
      MatrixXfR c_y = MatrixXfR::Zero(1, n_y);
      MatrixXfR m = MatrixXfR::Zero(n_x, n_y);
      MatrixXfR weights = MatrixXfR::Zero(n_x, n_y);
      for (int x = 0; x < n_x; ++x) {
        for (int y = 0; y < n_y; ++y) {
          weights(x, y) = inp(i, x, y);
        }
      }
      MinWeightedBipartiteCover(weights, &m, &c_x, &c_y);
      for (int x = 0; x < n_x; ++x) {
        cover_x(i, x, 0) = c_x(x, 0);
        for (int y = 0; y < n_y; ++y) {
          cover_y(i, 0, y) = c_y(0, y);
          matching(i, x, y) = m(x, y);
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Hungarian").Device(DEVICE_CPU), HungarianOp);
