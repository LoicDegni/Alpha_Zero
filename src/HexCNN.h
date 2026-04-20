#pragma once

#include <random>
#include <torch/torch.h>
#include <vector>



// ============================================================
// Bloc résiduel factorisé
// ============================================================
struct ResBlockImpl : torch::nn::Module {
    torch::nn::Conv2d      conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr},   bn2{nullptr};
 
    ResBlockImpl(int filters)
        : conv1(torch::nn::Conv2dOptions(filters, filters, 3).padding(1)),
          conv2(torch::nn::Conv2dOptions(filters, filters, 3).padding(1)),
          bn1(filters),
          bn2(filters)
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("bn1",   bn1);
        register_module("bn2",   bn2);
    }
 
    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        return torch::relu(x + residual);
    }
};
TORCH_MODULE(ResBlock);
 
// ============================================================
// Réseau principal
// ============================================================
struct HexCNNImpl : torch::nn::Module {
    static constexpr int NUM_RES_BLOCKS   = 5;
    static constexpr int NUM_FILTERS      = 32;
    static constexpr int INPUT_CHANNELS   = 2;   // J1 + J2
    static constexpr int VALUE_HIDDEN     = 128;
    static constexpr int POLICY_CHANNELS  = 2;
 
    int board_size;
 
    // --- Couche d'entrée ---
    torch::nn::Conv2d      conv_input{nullptr};
    torch::nn::BatchNorm2d bn_input{nullptr};
 
    // --- Corps partagé : blocs résiduels ---
    torch::nn::Sequential res_tower;
 
    // --- Tête de politique (Policy Head) ---
    torch::nn::Conv2d      conv_pol{nullptr};
    torch::nn::BatchNorm2d bn_pol{nullptr};
    torch::nn::Linear      fc_pol{nullptr};
 
    // --- Tête de valeur (Value Head) ---
    torch::nn::Conv2d      conv_val{nullptr};
    torch::nn::BatchNorm2d bn_val{nullptr};
    torch::nn::Linear      fc_val1{nullptr};
    torch::nn::Linear      fc_val2{nullptr};
 
    HexCNNImpl(int size)
        : board_size(size),
          // Entrée
          conv_input(torch::nn::Conv2dOptions(INPUT_CHANNELS, NUM_FILTERS, 3).padding(1)),
          bn_input(NUM_FILTERS),
          // Politique : conv 1×1 → POLICY_CHANNELS, puis Linear
          conv_pol(torch::nn::Conv2dOptions(NUM_FILTERS, POLICY_CHANNELS, 1)),
          bn_pol(POLICY_CHANNELS),
          fc_pol(POLICY_CHANNELS * (size + 4) * (size + 4), size * size),
          // Valeur : conv 1×1 → 1 canal, puis MLP
          conv_val(torch::nn::Conv2dOptions(NUM_FILTERS, 1, 1)),
          bn_val(1),
          fc_val1((size + 4) * (size + 4), VALUE_HIDDEN),
          fc_val2(VALUE_HIDDEN, 1)
    {
        register_module("conv_input", conv_input);
        register_module("bn_input",   bn_input);
 
        // Empiler les blocs résiduels
        for (int i = 0; i < NUM_RES_BLOCKS; ++i) {
            auto block = ResBlock(NUM_FILTERS);
            res_tower->push_back(block);
            register_module("res_block_" + std::to_string(i), block);
        }
        register_module("res_tower", res_tower);
 
        register_module("conv_pol", conv_pol);
        register_module("bn_pol",   bn_pol);
        register_module("fc_pol",   fc_pol);
 
        register_module("conv_val", conv_val);
        register_module("bn_val",   bn_val);
        register_module("fc_val1",  fc_val1);
        register_module("fc_val2",  fc_val2);
    }
 
    // [batch, 2, N+4, N+4] → { [batch, N*N], [batch, 1] }
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
 
        // --- Couche d'entrée ---
        x = torch::relu(bn_input->forward(conv_input->forward(x)));
 
        // --- Corps partagé (NUM_RES_BLOCKS blocs résiduels) ---
        x = res_tower->forward(x);
 
        // --- Tête de politique ---
        auto pol = torch::relu(bn_pol->forward(conv_pol->forward(x)));
        pol = pol.view({pol.size(0), -1});
        pol = fc_pol->forward(pol);            // logits bruts [batch, N*N]
 
        // --- Tête de valeur ---
        auto val = torch::relu(bn_val->forward(conv_val->forward(x)));
        val = val.view({val.size(0), -1});
        val = torch::relu(fc_val1->forward(val));
        val = torch::tanh(fc_val2->forward(val));  // [batch, 1]
 
        return std::make_tuple(pol, val);
    }
};
TORCH_MODULE(HexCNN);

// ============================================================
// Encoder le plateau en tenseur [1, 2, N+4, N+4].
// ============================================================
inline torch::Tensor encodeBoardState(const std::vector<char>& cells,
                                       int size, char currentPlayer) {
    int padded = size + 4;  // 2 de chaque côté
    auto tensor = torch::zeros({1, 2, padded, padded});
    auto acc = tensor.accessor<float, 4>();

    // --- Bordures : colonnes gauche/droite = joueur courant (canal 0) ---
    for (int r = 0; r < padded; r++) {
        acc[0][0][r][0] = 1.0f;
        acc[0][0][r][1] = 1.0f;
        acc[0][0][r][padded - 2] = 1.0f;
        acc[0][0][r][padded - 1] = 1.0f;
    }
    // --- Bordures : lignes haut/bas = adversaire (canal 1) ---
    for (int c = 0; c < padded; c++) {
        acc[0][1][0][c] = 1.0f;
        acc[0][1][1][c] = 1.0f;
        acc[0][1][padded - 2][c] = 1.0f;
        acc[0][1][padded - 1][c] = 1.0f;
    }

    // --- Plateau réel, décalé de +2 en r et c ---
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            char cell = cells[r * size + c];
            if (cell == '-') continue;
            int tr = (currentPlayer == 'X') ? r : c;
            int tc = (currentPlayer == 'X') ? c : r;
            int channel = (cell == currentPlayer) ? 0 : 1;
            acc[0][channel][tr + 2][tc + 2] = 1.0f;
        }
    }
    return tensor;
}

// ============================================================
// Inférence combinée : Politique et Valeur (Value)
// Retourne la distribution de politique ET la valeur du plateau
// ============================================================
inline std::pair<std::vector<float>, float> evaluateState(
        HexCNN& net,
        const std::vector<char>& cells,
        int size, char currentPlayer) {

    torch::NoGradGuard no_grad;
    net->eval();

    auto output = net->forward(encodeBoardState(cells, size, currentPlayer));
    auto logits = std::get<0>(output).squeeze(0).clone();  // [N*N]
    float value = std::get<1>(output).item<float>();       // Scalaire [-1, 1]

    // Masquer les cases occupées
    {
        auto acc = logits.accessor<float, 1>();
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++)
                if (cells[r * size + c] != '-') {
                    int rot = (currentPlayer == 'X') ? (r * size + c) : (c * size + r);
                    acc[rot] = -1e9f;
                }
    }

    auto probs_rot = torch::softmax(logits, 0);
    auto prob_acc  = probs_rot.accessor<float, 1>();

    // Retransformer dans l'espace original
    std::vector<float> probs(size * size, 0.0f);
    for (int r = 0; r < size; r++)
        for (int c = 0; c < size; c++) {
            int orig = r * size + c;
            int rot  = (currentPlayer == 'X') ? orig : (c * size + r);
            probs[orig] = prob_acc[rot];
        }
    return {probs, value};
}

// ============================================================
// Encodage de la politique de MCTS dans un tenseur [N*N].
// ============================================================
inline torch::Tensor encodePolicy(const std::vector< std::vector<int> > &visit_counts,
                                  int size, char currentPlayer, float totalVisits) {
    auto policy = torch::zeros({size * size});
    auto acc = policy.accessor<float, 1>();
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            int rot = (currentPlayer == 'X') ? (r * size + c) : (c * size + r);
            acc[rot] = visit_counts[r][c] / totalVisits;
        }
    }
    return policy;
}

// ============================================================
// Exemple d'entraînement
// ============================================================
struct TrainingExample {
    torch::Tensor state;   //[2, N+4, N+4]
    torch::Tensor policy;  // [N*N]
    float value_target;     // +1 ou -1 (Assigné en fin de partie)
    char player;           // Joueur courant au moment de cet état
};

// ============================================================
// Entraînement avec prise en compte de la politique et de la valeur
// ============================================================
inline std::tuple<float, float, float> trainOnBatch(HexCNN& net,
                           torch::optim::Optimizer& optimizer,
                           const std::vector<TrainingExample>& examples) {
    if (examples.empty()) return {0.0f, 0.0f, 0.0f};
    std::cerr << "test9\n";
    net->train();
    std::vector<torch::Tensor> states, policies, values;
    states.reserve(examples.size());
    policies.reserve(examples.size());
    values.reserve(examples.size());
    std::cerr << "test10\n";
    for (auto& ex : examples) {
        states.push_back(ex.state.unsqueeze(0));    //[1, 2, N+4, N+4]
        policies.push_back(ex.policy.unsqueeze(0)); //[1, N*N]
        values.push_back(torch::tensor({ex.value_target})); // [1, 1]
    }
    std::cerr << "test11\n";
    auto state_batch  = torch::cat(states,    0);
    auto policy_batch = torch::cat(policies,  0);
    auto value_batch  = torch::cat(values,    0);
    std::cerr << "test12\n";
    optimizer.zero_grad();
    auto output = net->forward(state_batch);
    auto logits = std::get<0>(output);
    auto values_pred = std::get<1>(output);

    // Cross-entropie pour la politique
    auto log_probs = torch::log_softmax(logits, 1);
    auto policy_loss = -(policy_batch * log_probs).sum(1).mean();
    std::cerr << "test13\n";
    // MSE (Erreur Quadratique Moyenne) pour la valeur
    auto value_loss = torch::mse_loss(values_pred, value_batch.unsqueeze(1));
    std::cerr << "test14\n";
    auto loss = policy_loss + value_loss; // On combine les deux erreurs
    loss.backward();
    optimizer.step();

    auto probs   = torch::softmax(logits.detach(), 1);
    auto entropy = -(probs * torch::log(probs + 1e-9f)).sum(1).mean();

    return {policy_loss.item<float>(), value_loss.item<float>(), entropy.item<float>()};
}

inline void entrainement(  HexCNN& net,
                    torch::optim::Optimizer& optimizer,
                    std::vector<TrainingExample> train_data,
                    unsigned int epochs, unsigned int batch_size,
                    std::mt19937& rng
                ) {
    std::cerr << "test9\n";
    for (unsigned int ep = 0; ep < epochs; ep++) {
        std::shuffle(train_data.begin(), train_data.end(), rng);
        float total_policy_loss = 0.0f;
        float total_value_loss = 0.0f;
        float total_entropy = 0.0f;
        int   batches       = 0;
        std::cerr << "test0\n";
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            size_t end = std::min(i + (size_t)batch_size, train_data.size());
            std::vector<TrainingExample> batch(
                train_data.begin() + i,
                train_data.begin() + end);

            std::cerr << "Test : " << i << "\n";
            auto[policy_loss, value_loss, entropy] = trainOnBatch(net, optimizer, batch);
            std::cerr << "Test* : " << i << "\n";
            total_policy_loss += policy_loss;
            std::cerr << "Test** : " << i << "\n";
            total_value_loss  += value_loss;
            std::cerr << "Test*** : " << i << "\n";
            total_entropy     += entropy;
            std::cerr << "Test**** : " << i << "\n";
            batches++;
            std::cerr << "Test***** : " << i << "\n";
        }
        std::cerr << "test11\n";
        std::cerr << "[train] Époque " << (ep + 1) << "/" << epochs
                    << "  policy_loss=" << (total_policy_loss / batches)
                    << "  value_loss="  << (total_value_loss / batches)
                    << "  entropy="     << (total_entropy / batches)
                    << std::endl;
    }
}

