#include <iostream>
#include <cstdio>
#include <csignal>
#include <chrono>
#include <thread>
#include <random>
#include <torch/torch.h>

#include "lib/CLI11.hpp"


#include "src/HexCNN.h"
#include "src/Hex_Environement.h"

#include "src/IA_Player.h"
#include "src/IANN_Player.h"
#include "src/ExternalProgram_Player.h"


bool noGUI=false;
void signal_handler(int signum) {
    if(!noGUI) {
        endwin();
    }
    exit(signum);
}

int main(int argc, char *argv[]) {
    std::signal(SIGINT, signal_handler);

    CLI::App app("Environnement pour le jeu Hex");

    std::string joueurX="";
    app.add_option("-X", joueurX, "Si -X \"IA\" alors le joueur X (rouge) sera joué par l'IA. Si -X \"IANN\" alors l'IA utilisera un réseau de neurones. Sinon, il s'agit d'un path vers un programme externe.");

    std::string joueurO="";
    app.add_option("-O", joueurO, "Si -O \"IA\" alors le joueur O (bleu) sera joué par l'IA. Si -O \"IANN\" alors l'IA utilisera un réseau de neurones. Sinon, il s'agit d'un path vers un programme externe.");

    bool slow=false;
    app.add_flag("--slow", slow, "Ajoutez un délai entre chaque coup de l'IA");

    app.add_flag("--noGUI", noGUI, "Désactiver l'affichage graphique");

    unsigned int taille=8;
    app.add_option("--size", taille, "Taille du plateau de jeu (default: 8)");

    unsigned int seed = 0;
    app.add_option("--seed", seed, "Seed pour l'aléatoire (par défaut: 0 pour seed aléatoire)");

    unsigned int mcts_simulation_time = 2000;
    app.add_option("--mcts-simulation-time", mcts_simulation_time, "Durée maximale pour les simulations MCTS par coup en ms (default: 2000)");

    // --- Options spécifiques pour AINN ---
    bool unactivate_value_head = false;
    app.add_flag("--unactivate-value-head", unactivate_value_head, "Désactiver la tête de valeur du CNN");

    std::string model_path = "";
    app.add_option("--model", model_path, "Chemin du modèle CNN à sauvegarder/charger (default: hex_cnn_SIZE.pt)");

    float C_puct = 2.5f;
    app.add_option("--Cpuct", C_puct, "Constante de contrôle de l'exploration dans MCTS (default: 2.5)");
    // --- Fin options spécifiques pour AINN ---

    // --- Options du mode entraînement ---
    bool train = false;
    app.add_flag("--train", train, "Mode entraînement : self-play + entraînement CNN");

    unsigned int train_games = 50;
    app.add_option("--train-games", train_games, "Nombre de parties de self-play par itération (default: 50)");

    unsigned int train_iterations = 1000;
    app.add_option("--train-iterations", train_iterations, "Nombre d'itérations (self-play + entraînement) (default: 1000)");

    float lr = 1e-3f;
    app.add_option("--lr", lr, "Taux d'apprentissage (default: 1e-3)");

    unsigned int epochs = 5;
    app.add_option("--epochs", epochs, "Nombre d'époques d'entraînement (default: 5)");

    unsigned int batch_size = 64;
    app.add_option("--batch-size", batch_size, "Taille des mini-lots (default: 64)");

    unsigned int replay_buffer_capacity = 20000;
    app.add_option("--replay-buffer-capacity", replay_buffer_capacity, "Capacité maximale du replay buffer (default: 20000)");  
    // --- Fin options mode entraînement ---


    CLI11_PARSE(app, argc, argv);

    if(seed==0) {
        seed = time(NULL);
    }
    srand(seed);

    if(model_path.size() == 0) {
        model_path = "hex_cnn_" + std::to_string(taille) + ".pt";   
    }

    // =========================================================================
    // MODE ENTRAÎNEMENT
    // =========================================================================
    if (train) {
        noGUI = true;

        HexCNN net(taille);

        // Charger un modèle existant si disponible
        try {
            torch::load(net, model_path);
            std::cerr << "[train] Modèle chargé depuis " << model_path << std::endl;
        } catch (...) {
            std::cerr << "[train] Nouveau modèle initialisé." << std::endl;
        }

        torch::optim::Adam optimizer(net->parameters(),
                                     torch::optim::AdamOptions(lr));

        std::mt19937 rng(seed);


        std::deque<TrainingExample> replay_buffer;

        // --- Boucle d'entraînement itérative ---
        for (unsigned int it = 0; it < train_iterations; it++) {
            std::cerr << "\n[train] ===== Itération " << (it + 1) << "/" << train_iterations << " =====" << std::endl;

            // --- Self-play : collecter les exemples de cette itération ---
            unsigned int new_examples_count = 0; // Pour compter les nouveaux ajouts

            for (unsigned int g = 0; g < train_games; g++) {
                std::vector<TrainingExample> game_examples;

                Hex_Environement hex(false, taille); 

                auto px = std::make_unique<IANN_Player>(net, 'X', taille, C_puct);
                auto po = std::make_unique<IANN_Player>(net, 'O', taille, C_puct);

                px->MCTS_TimeLimit(mcts_simulation_time);
                po->MCTS_TimeLimit(mcts_simulation_time);

                if(unactivate_value_head) {
                    px->unactivateValueHead();
                    po->unactivateValueHead();
                }
                
                px->enableDataCollection(&game_examples);
                po->enableDataCollection(&game_examples);

                hex.setPlayerX(std::move(px));
                hex.setPlayerO(std::move(po));

                // 1. Jouer la partie jusqu'à la fin
                while (!hex.isGameOver()) {
                    hex.play();
                }

                // 2. Récupérer le gagnant final
                char winner = hex.getWinner();

                // 3. Remplir la Value Head et envoyer dans le Replay Buffer
                for (auto& ex : game_examples) {
                    if (ex.player == winner) {
                        ex.value_target = 1.0f;
                    } else {
                        ex.value_target = -1.0f;
                    }
                    
                    replay_buffer.push_back(ex);
                }
                new_examples_count += game_examples.size();

                std::cerr << "[train] Partie " << (g + 1) << "/" << train_games
                          << "  gagnant=" << winner
                          << "  exemples_partie=" << game_examples.size() << std::endl;
            }

            if( new_examples_count < replay_buffer_capacity ) {
                while (replay_buffer.size() > replay_buffer_capacity) {
                    replay_buffer.pop_front();
                }
            }
            std::cerr << "[train] Exemples collectés cette itération : " << new_examples_count
                      << "  taille totale du replay buffer : " << replay_buffer.size() << "/" << replay_buffer_capacity << std::endl;

            if (replay_buffer.empty()) {
                std::cerr << "[train] Aucun exemple collecté, itération ignorée." << std::endl;
                continue;
            }
            
            // --- Entraînement sur les exemples ---
            entrainement( net, optimizer,
                   std::vector<TrainingExample>(replay_buffer.begin(), replay_buffer.end()),
                   epochs, batch_size, rng); 

            // --- Sauvegarder le modèle après chaque itération ---
            torch::save(net, model_path);
            std::cerr << "[train] Modèle sauvegardé dans " << model_path << std::endl;

            if( new_examples_count >= replay_buffer_capacity ) {
                replay_buffer.clear();
            }
        }

        return 0;
    }

    // =========================================================================
    // MODE JEU NORMAL
    // =========================================================================
    Hex_Environement hex(!noGUI, taille);

    if(joueurX.size()) {
        if(joueurX == "IA") {
            hex.setPlayerX(std::make_unique<IA_Player>('X', taille));
        } else if(joueurX == "IANN") {
            HexCNN net(taille);
            try {
                torch::load(net, model_path);
                std::cerr << "Modèle chargé depuis " << model_path << std::endl;
            } catch (...) {
                std::cerr << "Pas de modèle disponible à charger." << std::endl;
            }
            auto px = std::make_unique<IANN_Player>(net, 'X', taille, C_puct);

            px->MCTS_TimeLimit(mcts_simulation_time);

            if(unactivate_value_head) {
                px->unactivateValueHead();
            }
            hex.setPlayerX(std::move(px));
        } else {
            hex.setPlayerX(std::make_unique<ExternalProgram>(joueurX, 'X', taille));
        }
    } else {
        // Humain
    }

    if(joueurO.size()) {
        if(joueurO == "IA") {
            hex.setPlayerO(std::make_unique<IA_Player>('O', taille));
        } else if(joueurO == "IANN") {
            HexCNN net(taille);
            try {
                torch::load(net, model_path);
                std::cerr << "Modèle chargé depuis " << model_path << std::endl;
            } catch (...) {
                std::cerr << "Pas de modèle disponible à charger." << std::endl;
            }
            auto po = std::make_unique<IANN_Player>(net, 'O', taille, C_puct);

            po->MCTS_TimeLimit(mcts_simulation_time);

            if(unactivate_value_head) {
                po->unactivateValueHead();
            }
            hex.setPlayerO(std::move(po));
        } else {
            hex.setPlayerO(std::make_unique<ExternalProgram>(joueurO, 'O', taille));
        }
    } else {
        // Humain
    }

    hex.CNN_for_visualization(model_path);

    while(hex.isGameOver() == false) {
        hex.printBoard();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        hex.play();
        if(noGUI && ((hex.getPlayer() == 'O' && joueurX == "IA") || (hex.getPlayer() == 'X' && joueurO == "IA"))) {
            std::cout << std::get<0>(hex.getLastMove()) << " " << std::get<1>(hex.getLastMove()) << std::endl;
        }
        if(slow) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    hex.afficherFin();

    if(!noGUI) {
        while(getch() != 10) {
            hex.afficherFin();
            std::cerr << "Appuyer sur ENTER ou CTRL+C pour quitter" << std::endl;
        }
    } else {
        std::cerr << "Le joueur " << hex.getWinner() << " a gagné" << std::endl;
    }

    return 0;
}
