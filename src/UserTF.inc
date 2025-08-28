// UserTF.inc
#pragma once

// Implementations for UserTF declared in UserTF.h

// ----------------- Helpers -------------------

// Creates a TF variable of given shape, stores it in bookkeeping arrays.
Variable UserTF::MakeVar(const std::string &name, const TensorShape &shape) {
    Variable v(root_.WithOpName(name), shape, DT_FLOAT);
    vars_.push_back(v);
    var_shapes_.push_back(shape);
    return v;
}

// Initializes weights with truncated normal, scaled by init_std
Output UserTF::WeightInit(int in, int out) {
    return Multiply(root_, TruncatedNormal(root_, {in, out}, DT_FLOAT), Const(root_, hp_.init_std));
}

// Biases initialized to zeros
Output UserTF::BiasInit(int dim) {
    return Fill(root_, Const(root_, {dim}), Const(root_, 0.0f));
}

// Dense helper: linear -> activation (ReLU or leaky ReLU if leaky==true)
// Dense layer = in * W + b. Uses ReLU or Leaky ReLU activation.
Output UserTF::Dense(Output in, Variable &W, Variable &b, bool leaky) {
    auto lin = Add(root_, MatMul(root_, in, W), b);
    if (!leaky) return Relu(root_, lin);
    auto alpha = Const(root_, 0.01f);
    return Maximum(root_, lin, Multiply(root_, lin, alpha));
}

// Prepare Adam slot Variables and their initializer ops (init_slots_)
void UserTF::PrepareAdamSlots() {
    m_slots_.clear();
    v_slots_.clear();
    init_slots_.clear();

    m_slots_.reserve(vars_.size());
    v_slots_.reserve(vars_.size());

    for (size_t i = 0; i < vars_.size(); ++i) {
    m_slots_.emplace_back(root_, var_shapes_[i], DT_FLOAT);
    v_slots_.emplace_back(root_, var_shapes_[i], DT_FLOAT);
    init_slots_.push_back(Assign(root_, m_slots_[i], ZerosLike(root_, m_slots_[i])).operation);
    init_slots_.push_back(Assign(root_, v_slots_[i], ZerosLike(root_, v_slots_[i])).operation);
    }
    // beta powers
    init_slots_.push_back(Assign(root_, beta1_power_, Const(root_, hp_.beta1)).operation);
    init_slots_.push_back(Assign(root_, beta2_power_, Const(root_, hp_.beta2)).operation);
}

// Create Adam Apply ops (populate adam_ops_) from gradients (grads vector)
void UserTF::CreateAdamUpdates(const std::vector<Output> &grads) {
    adam_ops_.clear();
    adam_ops_.reserve(vars_.size());
    for (size_t i = 0; i < vars_.size(); ++i) {
    auto apply = ApplyAdam(
        root_,
        vars_[i],
        m_slots_[i],
        v_slots_[i],
        beta1_power_,
        beta2_power_,
        Const(root_, hp_.learning_rate),
        Const(root_, hp_.beta1),
        Const(root_, hp_.beta2),
        Const(root_, hp_.epsilon),
        grads[i]
    );
    adam_ops_.push_back(apply.operation);
    }
}

// ----------------- End Helpers -------------------

// ---------------- BuildModel -----------------
// Constructs placeholders, variables, forward pass, loss, gradients, and Adam ops.
Status UserTF::BuildModel() {
    // Basic checks
    if (hp_.input_dim <= 0) return tensorflow::errors::InvalidArgument("input_dim must be > 0");
    if (hp_.output_dim <= 0) return tensorflow::errors::InvalidArgument("output_dim must be > 0");
    for (auto h : hp_.hidden_layers) if (h <= 0) return tensorflow::errors::InvalidArgument("hidden layer sizes must be > 0");

    // placeholders
    X_ph_ = Placeholder(root_.WithOpName("X"), DT_FLOAT,
                        Placeholder::Shape({hp_.batch_size, hp_.input_dim}));
    Y_ph_ = Placeholder(root_.WithOpName("Y"), DT_FLOAT,
                        Placeholder::Shape({hp_.batch_size, hp_.output_dim}));

    // layer sizes: [input, hidden..., output]
    std::vector<int> layer_sizes;
    layer_sizes.push_back(hp_.input_dim);
    for (auto h : hp_.hidden_layers) layer_sizes.push_back(h);
    layer_sizes.push_back(hp_.output_dim);

    // create variables: W0,b0, W1,b1, ... (store in vars_ via MakeVar)
    const int n_layers = static_cast<int>(layer_sizes.size()) - 1;
    vars_.clear();
    var_shapes_.clear();
    for (int i = 0; i < n_layers; ++i) {
        std::string wname = "W" + std::to_string(i+1);
        std::string bname = "b" + std::to_string(i+1);
        TensorShape wshape({layer_sizes[i], layer_sizes[i+1]});
        TensorShape bshape({layer_sizes[i+1]});
        MakeVar(wname, wshape);
        MakeVar(bname, bshape);
    }

    // prepare init_ops_ (assign weight_init or bias_init depending on index)
    init_ops_.clear();
    for (size_t i = 0; i < vars_.size(); ++i) {
        if (i % 2 == 0) {
            // weight
            init_ops_.push_back(
            Assign(root_, vars_[i], WeightInit(var_shapes_[i].dim_size(0), var_shapes_[i].dim_size(1))).operation
            );
        } else {
            // bias
            init_ops_.push_back(
            Assign(root_, vars_[i], BiasInit(var_shapes_[i].dim_size(0))).operation
            );
        }
    }

    // forward pass: chain through variables in pairs (W,b)
    Output current = X_ph_;
    for (int layer = 0; layer < n_layers; ++layer) {
        Variable &W = vars_[2*layer + 0];
        Variable &b = vars_[2*layer + 1];
        bool is_last = (layer == n_layers - 1);
        // example rule: use leaky ReLU on odd-indexed hidden layers
        bool use_leaky = (layer % 2 == 1);
        if (is_last) {
            current = Add(root_, MatMul(root_, current, W), b); // linear output
        } else {
            current = Dense(current, W, b, use_leaky);
        }
    }
    output_ = current;

    // loss: mean squared error
    auto sq_diff = Square(root_, Sub(root_, output_, Y_ph_));
    loss_ = Mean(root_, sq_diff, {0,1});

    // compute gradients for all variables
    std::vector<Output> var_outputs;
    var_outputs.reserve(vars_.size());
    for (auto &v : vars_) var_outputs.push_back(v);

    std::vector<Output> grads;
    Status s = AddSymbolicGradients(root_, {loss_}, var_outputs, &grads);
    if (!s.ok()) return s;

    // Prepare Adam slots and create updates
    PrepareAdamSlots();
    CreateAdamUpdates(grads);

    return tensorflow::OkStatus();
}
// ---------------- End BuildModel -----------------

// ---------------- Initialize -----------------
Status UserTF::Initialize() {
    // initialize model variables
    TF_RETURN_IF_ERROR(sess_.Run({}, {}, init_ops_, nullptr));
    // initialize Adam slots if present
    if (!init_slots_.empty()) {
        TF_RETURN_IF_ERROR(sess_.Run({}, {}, init_slots_, nullptr));
    }
    return tensorflow::OkStatus();
}
// ---------------- End Initialize -----------------

// --------------- TrainOnBatch ----------------
// Executes one training step (compute loss + apply Adam updates)
// Returns the loss value if out_loss provided.
Status UserTF::TrainOnBatch(const Tensor &X_batch, const Tensor &Y_batch, float *out_loss) {
    if (adam_ops_.empty()) return tensorflow::errors::FailedPrecondition("adam_ops_ is empty");
    std::vector<Tensor> outputs;
    TF_RETURN_IF_ERROR(sess_.Run({{X_ph_, X_batch}, {Y_ph_, Y_batch}}, {loss_}, adam_ops_, &outputs));
    if (out_loss) *out_loss = outputs[0].scalar<float>()();
    return tensorflow::OkStatus();
}
// --------------- End TrainOnBatch ----------------

// ---------------- Train (template) ----------------
template <typename BatchGen>
Status UserTF::Train(BatchGen batch_generator) {
    TF_RETURN_IF_ERROR(Initialize());
    for (int epoch = 0; epoch < hp_.num_epochs; ++epoch) {
        Tensor Xb(DT_FLOAT, TensorShape({hp_.batch_size, hp_.input_dim}));
        Tensor Yb(DT_FLOAT, TensorShape({hp_.batch_size, hp_.output_dim}));
        if (!batch_generator(Xb, Yb, epoch)) break;
        float loss = 0.0f;
        TF_RETURN_IF_ERROR(TrainOnBatch(Xb, Yb, &loss));
        if (epoch % 50 == 0) std::cout << "Epoch " << epoch << "  Loss=" << loss << std::endl;
    }
    std::cout << "Training complete." << std::endl;
    return tensorflow::OkStatus();
}
// ---------------- End Train (template) ----------------

// ---------------- MakeSyntheticBatch ----------------
// Generates toy nonlinear data for training/testing
void UserTF::MakeSyntheticBatch(Tensor &Xb, Tensor &Yb) {
    auto xb = Xb.matrix<float>();
    auto yb = Yb.matrix<float>();
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < hp_.batch_size; ++i) {
        float x1 = dist(rng_), x2 = dist(rng_), x3 = dist(rng_);
        // guard for input_dim > 3 (fill remaining with 0)
        xb(i,0)=x1; if (hp_.input_dim > 1) xb(i,1)=x2; if (hp_.input_dim > 2) xb(i,2)=x3;
        for (int j = 3; j < hp_.input_dim; ++j) xb(i,j) = 0.0f;
        // compute y using original formula (works for output_dim==1)
        if (hp_.output_dim == 1) {
            yb(i,0)=std::sin(x1)*std::exp(x2) + std::tanh(x3*x3);
        } else {
            // if output_dim>1 fill first entry with the function and others with 0
            yb(i,0)=std::sin(x1)*std::exp(x2) + std::tanh(x3*x3);
            for (int k = 1; k < hp_.output_dim; ++k) yb(i,k) = 0.0f;
        }
    }
}
// ---------------- End MakeSyntheticBatch ----------------

// ---------------- Predict ----------------
// Single forward pass prediction
Status UserTF::Predict(const std::vector<float> &input, std::vector<float> *output_vec) {
    if ((int)input.size() != hp_.input_dim) return tensorflow::errors::InvalidArgument("input mismatch");
    Tensor Xb(DT_FLOAT, TensorShape({1, hp_.input_dim}));
    auto xb = Xb.matrix<float>();
    for (int i = 0; i < hp_.input_dim; ++i) xb(0,i) = input[i];
    std::vector<Tensor> outputs;
    TF_RETURN_IF_ERROR(sess_.Run({{X_ph_, Xb}}, {output_}, &outputs));
    auto outm = outputs[0].matrix<float>();
    output_vec->resize(hp_.output_dim);
    for (int i = 0; i < hp_.output_dim; ++i) (*output_vec)[i] = outm(0,i);
    return tensorflow::OkStatus();
}
// ---------------- End Predict ----------------
