// UserTF.h
#pragma once

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/lib/core/errors.h"

// ******** Definition namespace for TF module ******** //
namespace tf_cpp_model {

using namespace tensorflow;
using namespace tensorflow::ops;
// Simple hyper-parameter container (modifiable by caller)
struct HyperParams {
    int input_dim   = 3;
    // dynamic list of hidden dims
    std::vector<int> hidden_layers = {128, 256, 128, 64, 32};
    int output_dim  = 1;
    int batch_size  = 128;          // training configurations
    int num_epochs  = 500;
    float init_std  = 0.02f;        // stddev for weight initiation
    float learning_rate = 0.0005f;  // Adam optimizer parameters
    float beta1     = 0.9f;
    float beta2     = 0.999f;
    float epsilon   = 1e-8f;
};

// ******** Definition of UserTF Class ******** //
class UserTF {
private:
    // config + graph/session
    HyperParams hp_;
    Scope root_;            // TensorFlow computation graph builder scope.
    ClientSession sess_;    // Execution engine to run ops.
    // RNG for example data
    std::mt19937 rng_;
    // Model bookkeeping
    std::vector<Variable> vars_;            // W0,b0, W1,b1, ...
    std::vector<TensorShape> var_shapes_;   // variable dimensions
    // Optional Adam slots/powers
    std::vector<Variable> m_slots_;
    std::vector<Variable> v_slots_;
    Variable beta1_power_;                  // track exponential decay powers
    Variable beta2_power_;
    // Ops
    std::vector<Operation> init_ops_;
    std::vector<Operation> init_slots_;
    std::vector<Operation> adam_ops_;
    // Placeholders & outputs (set when building the model)
    Output X_ph_;
    Output Y_ph_;
    Output output_;                         // network predictio
    Output loss_;                           // mean squared error

public:
    // constructor copies hyperparameters, creates root scope + session
    explicit UserTF(const HyperParams &hp)
    :   hp_(hp),
        root_(Scope::NewRootScope()),
        sess_(root_),
        rng_(42),
        beta1_power_(root_, TensorShape({}), DT_FLOAT),
        beta2_power_(root_, TensorShape({}), DT_FLOAT)
    { }
    ~UserTF(){ };

    // Build the model graph using hp_. This constructs placeholders, variables,
    // forward pass (output_), loss_, gradients and optimizer ops (adam_ops_).
    // Call BuildModel() once before Initialize().
    Status BuildModel();

    // Initialize variables (weights/biases) and Adam slots/powers.
    Status Initialize();

    // Single training step on provided batch. Returns loss through out_loss.
    Status TrainOnBatch(const Tensor &X_batch, const Tensor &Y_batch, float *out_loss = nullptr);

    // Generic training loop. BatchGen fills Xb/Yb and returns true to continue.
    // Template implementation must be provided in tf_regressor.inc
    template <typename BatchGen>
    Status Train(BatchGen batch_generator);

    // Utility: create synthetic batch (same function as original example)
    void MakeSyntheticBatch(Tensor &Xb, Tensor &Yb);

    // Single-example prediction (input size == input_dim)
    Status Predict(const std::vector<float> &input, std::vector<float> *output_vec);

private:
    // Encapsulates repeated tasks: creating variables, initializing weights/biases,
    // building layers, setting up Adam slots/updates.

    // Helpers for building the graph: use MakeVar() inside BuildModel().
    Variable MakeVar(const std::string &name, const TensorShape &shape);
    Output WeightInit(int in, int out);
    Output BiasInit(int dim);

    // Small dense helper (declare here; implement in .inc)
    Output Dense(Output in, Variable &W, Variable &b, bool leaky = false);

    // Internal helpers to create Adam slots & update ops (bodies in .inc)
    void PrepareAdamSlots();
    void CreateAdamUpdates(const std::vector<Output> &grads);

};

// Include implementation file (method bodies).
#include "src/UserTF.inc"
}
