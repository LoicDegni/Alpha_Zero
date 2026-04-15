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

class IANN_Player : public Player_Interface {
    HexCNN _net;
    char _player;
    unsigned int _taille;
    unsigned int _time_limit_ms = 2000; // Par défaut, 2 secondes par coup
    std::vector< std::tuple<unsigned int, unsigned int, char> > _historique_coups;
    float _C_puct;
    std::mt19937 _rng;

    struct Node {
        Node* parent= nullptr;
        std::vector<Node*> children;
        int moveRow, moveCol;
        char playerJustMoved;
        int visits = 0;
        double wins = 0;

        std::vector<float> _politique;
        std::vector<int> toVisit;
        std::vector<int> untriedMoves;
    };
    Node* _root = nullptr;

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
         *  Upper Confidence Trees (UCT)
         *  RAVE (Rapid Action Value Estimation)
        */
        double C = 1.414;
        Node* best = nullptr;
        double bestValue = -1e9;

        for(auto child: node->children) {
            double exploitation_S_i = child->wins / (child->visits);
            double exploration_S_i = C * sqrt(log(node->visits) / (child->visits));
            //On previent le cas ou child->rave_visits = 0
            double rave_ratio = child->rave_wins/(child->rave_visits +1e-6);
            double w = ( child->rave_visits/(child->visits + child->rave_visits + 1e-6) );
            double score = ((1 - w)*exploitation_S_i) + (w * rave_ratio) + exploration_S_i; 
            if (score > bestValue) 
            {
                bestValue = score;
                best = child;
            }
        }
        _uf.applyMoveUF(best->moveRow, best->moveCol, best->playerJustMoved);
        return best;
    }
   
    Node* expand(Node* node) {
        /**
         * Fonction qui recoit un noeud courant, recupere un mouvement possible
         * du noeud et creer un noeud enfant avec le mouvement recuperé
         * 
         * Return:      Le noeud enfant
        */
        int moveID;
        Node* child = new Node();
        
        moveID = node->untriedMoves.back();        
        node->untriedMoves.pop_back();

        child->parent = node;
        child->moveRow = convertIDToCoordonate(moveID).first;
        child->moveCol = convertIDToCoordonate(moveID).second;
        child->playerJustMoved = (node->playerJustMoved == 'X') ? 'O' : 'X';

        child->toVisit = node->toVisit;

        //maj de child->tovisit
        auto it = std::find(child->toVisit.begin(), child->toVisit.end(),moveID);
        if (it != child->toVisit.end()) {
            std::swap(*it,child->toVisit.back());
            child->toVisit.pop_back();
        }
        child->untriedMoves = child->toVisit;
        child->_politique = _net.evaluateState();
        node->children.push_back(child);

        // On met a jour la carte _uf[O(n)]
        _uf.applyMoveUF(child->moveRow, child->moveCol, child->playerJustMoved);
        return child;
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
        raveSimulationUpdate(node, played_moves, pl);
        return pl;
    }

    void backpropagate(Node* node, char winner) {
        /**
         * La fonction remonte l'arbre MCTS et mets à jour les noeuds.
        */
       while (node != nullptr) {
        node->visits++;

        if (node->playerJustMoved == winner){
            node->wins++;
        }
        node = node->parent;
       }
    }
//-------------------ALGO MCTS-------------------//
public:
// Le constructeur par défaut avec HexCNN en paramètre
    IANN_Player(const HexCNN& net, char player, unsigned int taille=10, float Cpuct=2.5f) 
        : _net(net), _player(player), _taille(taille), _C_puct(Cpuct), _rng(std::random_device{}()) {
        assert(player == 'X' || player == 'O');
    }

    void otherPlayerMove(int row, int col) override {
        // l'autre joueur à joué (row, col)
    }

    std::tuple<int, int> getMove(Hex_Environement& hex) override {
        int row, col; // TODO TP4 : choisir le coups row, col a jouer

        ///// Exemple d'un choix aléatoire ////////////
        do {
            row = rand()%_taille; 	// Choix aléatoire
            col = rand()%_taille; 	// Choix aléatoire
        } while( hex.isValidMove(row, col) == false );
        //////////////////////////////////////////////

        return {row, col};
    }

    void MCTS_TimeLimit(unsigned int time_limit_ms) {
        _time_limit_ms = time_limit_ms;
    }

    void unactivateValueHead() {
        // TODO TP4 : désactiver la tête de valeur du CNN
    }

    void enableDataCollection(std::vector<TrainingExample>* examples) {
        // TODO TP4 : activer la collecte de données pour l'entraînement du CNN
    }

private:
    Node* FindBestChild(Node* node) {
        /**
         * Retourne le noeud le plus prometteurs
         * 
         * Le noeud le plus visité. Lorsque plusieurs noeud
         * ont le même nombre de visite, on compare leurs 
         * nombre de victoire
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


};
