#pragma once

#include <tuple>
#include <vector>
#include <memory>
#include <cassert>
#include <limits>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>


#include "HexCNN.h"
#include "Hex_Environement.h"
#include "IA_Player.h"


class IANN_Player : public Player_Interface {
    HexCNN _net;
    char _player;
    unsigned int _taille;
    unsigned int _time_limit_ms = 2000; // Par défaut, 2 secondes par coup
    std::vector<char> _board;
    
    std::vector< std::tuple<unsigned int, unsigned int, char> > _historique_coups;
    
    bool _unactivate_value_head = false;
    bool _training_mode = false;
    
    float _C_puct;
    
    UnionFind _uf;
    std::mt19937 _rng;

    struct Node {
        Node* parent= nullptr;
        std::vector<Node*> children;
        int moveRow, moveCol;
        char playerJustMoved;

        int visits = 0;
        float valueSum = 0.0;
        float Apriori = 0.0;

        std::vector<float> politique;
        std::vector<int> toVisit;
        std::vector<int> untriedMoves;

        bool expanded = false;
    };
    Node* _root = nullptr;

    std::vector<TrainingExample>* _training_examples = nullptr;

    std::vector<float> get_dirichlet_noise(int taille_tableau, float alpha = 0.3f ) {
        // Génération du bruit de Dirichlet via des lois Gamma
        std::gamma_distribution<float> gamma(alpha, 1.0f);
        std::vector<float> noise(taille_tableau);
        float sum = 0.0f;
        for (size_t k = 0; k < noise.size(); ++k) {
            noise[k] = gamma(_rng);
            sum += noise[k];
        }

        // Normalisation pour que la somme soit égale à 1
        for (size_t k = 0; k < noise.size(); ++k) {
            noise[k] /= sum;
        }

        return noise;
    }

//-------------------ALGO MCTS-------------------//
    Node* select(Node* node) {
        /**
         * La fonction selectionne le noeud le plus prometteur
         * parmis tous les enfants du noeud courant.
         * Stratégie:
         * PUCT (Predictor Upper Confidence bounds applied to Trees)
        */
        double C = 2.5;
        Node* best = nullptr;
        double bestValue = -1e9;

        double exploitation_S_i = 0;
        double exploration_S_i = 0;

        for(auto child: node->children) {
            if (child->visits > 0) {
                exploitation_S_i = child->valueSum / (child->visits);
            }else {
                exploitation_S_i = 0;
            }
            exploration_S_i = C * (sqrt(node->visits) /(1 + (child->visits))) * child->Apriori;
            double score =  exploration_S_i + exploitation_S_i;
            if (score > bestValue)
            {
                bestValue = score;
                best = child;
            }
        }
        _uf.applyMoveUF(best->moveRow, best->moveCol, best->playerJustMoved);
        return best;
    }

    float expand(Node* node) {
        /**
         * Fonction qui recoit un noeud courant, recupere un mouvement possible
         * du noeud et creer un noeud enfant avec le mouvement recuperé
         * 
         * Return:          Le noeud enfant
         */
         
        auto [politiques, value] = evaluateState(_net, _board, _taille, _player);
        node->politique = politiques;

        while(!node->untriedMoves.empty()){
            int moveID = node->untriedMoves.back();        
            node->untriedMoves.pop_back();
            
            Node* child = new Node();
            child->Apriori = politiques[moveID];
            child->visits = 0;
            child->valueSum = 0;
            child->parent = node;
            child->moveRow = convertIDToCoordonate(moveID).first;
            child->moveCol = convertIDToCoordonate(moveID).second;
            child->playerJustMoved = (node->playerJustMoved == 'X') ? 'O' : 'X';

            //maj de child->tovisit
            auto it = std::find(child->toVisit.begin(), child->toVisit.end(),moveID);
            if (it != child->toVisit.end()) {
                std::swap(*it,child->toVisit.back());
                child->toVisit.pop_back();
            }
            
            child->untriedMoves = child->toVisit;
            node->children.push_back(child);
            }
        node->expanded = true;
        return value;
    }

    char simulate(Node* node) {
        /**
         * La fonction simule toute la suite de la partie 
         * du noeud courant, mets à jour les variables
         * et retourne le gagnant.
        */
        std::vector<int> played_moves;
        char pl = node->playerJustMoved;

        if (node->toVisit.empty()) {
            return node->playerJustMoved;
        }
        simulateToTheEnd(pl,node->toVisit, played_moves);
        return pl;
    }

    void backpropagateActivatedVH (Node* node, float v) {
        /**
         * La fonction remonte l'arbre MCTS et mets à jour les noeuds.
        */
        while (node != nullptr) {
            node->visits++;
            node->valueSum += v;  
            v = -v;
            node = node->parent;
        }
    }

    void backpropagateUnactivatedVH (Node* node, char winner) {
        /**
         * La fonction remonte l'arbre MCTS et mets à jour les noeuds.
        */
       while (node != nullptr) {
        node->visits++;

        if (node->playerJustMoved == winner){
            node->valueSum++;
        }
        node = node->parent;
       }
    }
//-------------------ALGO MCTS-------------------//

public:
// Le constructeur par défaut avec HexCNN en paramètre
    IANN_Player(const HexCNN& net, char player, unsigned int taille=10, float Cpuct=2.5f) 
        : _net(net), _player(player), _taille(taille), _C_puct(Cpuct), _rng(std::random_device{}()), _uf(taille) {
        assert(player == 'X' || player == 'O');
    }

    void otherPlayerMove(int row, int col) override {
        /**
         * Lorsque l'agent le coup du joueur, 
         * il met à jour sont historique interne et 
         * avance dans l'arbre avec l'etat courant. 
         * Si il ne trouve pas de noeud dans son arbre qui correspond
         * au coup joué par l'adversaire, il creer une nouvelle racine 
         * avec l'etat courant
        */
        _historique_coups.push_back({row, col, (_player == 'X') ? 'O' : 'X'});
        _board[convertCoordonateToID(row,col)] = (_player == 'X') ? 'O' : 'X';

        if(_root != nullptr) {
            for(auto child : _root->children) {
                if(child->moveRow == row && child->moveCol == col) {
                    _root = child;
                    _root->parent = nullptr;
                    // On met a jour la carte _uf[O(n)]
                    resetUFToNow();
                    return;
                }
            }
            _root = nullptr; 
            // On met a jour la carte _uf[O(n)]
            resetUFToNow();
        }
        //auto [probs, value] = evaluateState(_net, _board, _taille, (_player == 'X') ? 'O' : 'X');
        //_root->politique 
    }

    std::tuple<int, int> getMove(Hex_Environement& hex) override {
        auto start = std::chrono::steady_clock::now();
        auto deadline = start + std::chrono::milliseconds(_time_limit_ms);

        if(_root == nullptr) {
            std::cerr << "test1\n";
            _root = new Node();
            _root->playerJustMoved = (_player == 'X') ? 'O' : 'X';
            getAllMoves(hex);
        }
        std::cerr << "test2\n";
        auto [probs, value] = evaluateState(_net, _board, _taille, _player);
        _root->politique = probs;
        std::cerr << "test3\n";

        if (_training_mode) {
            float epsilon = 0.25f;
            float alpha = 0.1f * (_taille * _taille);
            auto noise = get_dirichlet_noise(_taille * _taille, alpha);
            for (int i = 0; i < _taille * _taille; i++) {
                _root->politique[i] = (1 - epsilon) * _root->politique[i] + epsilon * noise[i];
            }
        }

        while (std::chrono::steady_clock::now() < deadline) {
            Node* node = _root;
            float value;
            char current_player;
            char winner;

            // 1. Sélection
            while(node->expanded && !node->children.empty())
                node = select(node);
            
            // 2. Expansion
            //if(!node->untriedMoves.empty())
            value = expand(node);
            
            // 3. Simulation
            if (_unactivate_value_head) {
                if (!_uf.hasWinner(node->playerJustMoved)) 
                    winner = simulate(node);
                else
                    winner = node->playerJustMoved;
            }

            // 4. Rétropropagation
            if(_unactivate_value_head)
                backpropagateActivatedVH(node, value);
            else
                backpropagateUnactivatedVH(node, winner);
            resetUFToNow();
        }
        Node* best;
        if (_training_mode) {
            best = SampleBestChild(_root);
        } else {
            best = FindBestChild(_root);
        }

        std::vector<std::vector<int>> visit_counts(_taille, std::vector<int>(_taille, 0));
        float totalVisits = 0.0f;

        for (Node* child : _root->children) {
            int r = child->moveRow;
            int c = child->moveCol;

            visit_counts[r][c] = child->visits;
            totalVisits += child->visits;
        }

        // Coup joué
        if (_training_mode && _training_examples != nullptr)
        {
            TrainingExample example;

            // 1. Etat
            example.state = encodeBoardState(_board, _taille, _player);

            // 2. Politique (issue du MCTS)
            example.policy = encodePolicy(visit_counts, _taille, _player, totalVisits);

            // 3. Joueur courant
            example.player = _player;

            // 4. IMPORTANT : laisser à 0 (rempli plus tard par main.cpp)
            example.value_target = 0.0f;

            // 5. Ajout au dataset
            _training_examples->push_back(example);
        }

        _board[convertCoordonateToID(best->moveRow, best->moveCol)] = _player;
        _historique_coups.push_back({best->moveRow, best->moveCol, _player});
        _root = best;
        _root->parent = nullptr;
        
        return {best->moveRow, best->moveCol};
    }

    void MCTS_TimeLimit(unsigned int time_limit_ms) {
        _time_limit_ms = time_limit_ms;
    }

    void unactivateValueHead() {
        /**
         * Fonction qui force le MCTS à utiliser 
         * des rollouts aléatoires 
        */
       _unactivate_value_head = true;
    }

    void enableDataCollection(std::vector<TrainingExample>* examples) {
        /**
         * Fonction qui active le mode entrainement. 
        */
        _training_mode = true;
        _training_examples = examples;
    }

private:
    void getAllMoves(Hex_Environement& hex) {
        /**
         * Fonction qui recupere tout les coups valides restant
         * dans la partie et les mets à jours au noeud racine.
        */
        for(unsigned int i=0; i < _taille; i++) {
            for(unsigned int j = 0; j< _taille; j++) {
                if(hex.isValidMove(i,j)) {
                    _root->toVisit.push_back(convertCoordonateToID(i,j));
                    _root->untriedMoves.push_back(convertCoordonateToID(i,j));
                }
            }
        }
    }

    void simulateToTheEnd(char& pl, std::vector<int>& available_moves, std::vector<int>& played_moves){
        /**
         * Fonction qui simule une partie jusqu'a ce qu'il y ai un gagnant. 
         * La structure unionFind(_uf) simule l'etat du jeu
        */
        do {
            pl = (pl == 'X') ? 'O' : 'X';
            std::uniform_int_distribution<int> uniform_moves_distribution(0, available_moves.size() -1);
            int random_index = uniform_moves_distribution(_rng);
            auto id = available_moves[random_index];
            auto move = convertIDToCoordonate(id);
            played_moves.push_back(id);
            _uf.applyMoveUF(move.first, move.second, pl);
        }while (!_uf.hasWinner(pl));

        if (!_uf.hasWinner('X') && !_uf.hasWinner('O')){
            std::cerr << "Erreur: available list est vide\n";
            std::exit(EXIT_FAILURE);
        }
    }

    Node* FindBestChild(Node* node) {
        /**
         * Retourne le noeud le plus prometteur
         * 
         * Le noeud le plus visité.
        */
        Node* best = nullptr;
        int maxVisits = -1;

        for (auto child : node->children) {
            if (child->visits > maxVisits) {
                maxVisits = child->visits;
                best = child;
            }
        }
        return best;
    }

    Node* SampleBestChild(Node* node) {
        std::vector<double> probs;
        double sum = 0.0;

        for (auto child : node->children) {
            double p = child->visits;
            probs.push_back(p);
            sum += p;
        }
        if (sum == 0) {
            return node->children[rand() % node->children.size()];
        }

        // normalisation
        for (auto& p : probs) {
            p /= sum;
        }

        // tirage
        double r = (double)rand() / RAND_MAX;
        double acc = 0.0;

        for (size_t i = 0; i < node->children.size(); i++) {
            acc += probs[i];
            if (r <= acc) {
                return node->children[i];
            }
        }

        return node->children.back();
    }

    int distanceToCenter(int r, int c, int N) {
        /**
         * Fonction calcul la distance de Manhattan 
         * d'une position au centre de la table de jeu
        */
        int center = N / 2;
        return std::abs(r - center) + std::abs(c - center);
    }

    int convertCoordonateToID(int r, int c) {
        /**
         * Fonction qui convertit une coordonne(r,c)
         * en un identifiant unique
        */
        return r * _taille + c;
    }

    std::pair<int, int> convertIDToCoordonate(int id) {
        /**
         * Fonction qui convertit un identifiant 
         * en sa coordonnée(r,c) d'origine
        */
       return {id / _taille, id % _taille};
    }

    void resetUFToNow(){
        /**
         * Fonction qui remet à zéro la structure unionFind
         * et ensuite la mets à jours avec l'historiques de
         * coups courant.
        */
        _uf.reset();
        for(const auto& [r,c,pl]: _historique_coups) {
            _uf.applyMoveUF(r,c,pl);
            }
    }

//====TP4====

};
