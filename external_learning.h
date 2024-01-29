/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   external_learning.h
 * Author: ron
 *
 * Created on November 22, 2022, 10:07 AM
 */

#ifndef EXTERNAL_LEARNING_H
#define EXTERNAL_LEARNING_H

#define CEG
#define COMPACT

#include <iostream>
#include <cassert>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#include <cstring>
#include <string>
#include <cmath>
#include <math.h>
#include <limits>
#include <algorithm>


/**
 * Simple implementation of a Q-learning algorithm
 * This implementation is *NOT* intended to be efficient, rather it is
 * implemented in the most straight-forward manner to illustrate how to
 * correctly use the external-learning functionality of Uppaal.
 *
 * Notice that we truncate concrete state-values to nearest integer to avoid
 * and explosion in the Q-table.
 */
class QLearner {
private:

    const double min_reward = -32767.0;
    /**
     * Struct for handling the Q-value update
     */
    struct qvalue_t {
        double _value = 0;
        size_t _count = 0;
#ifdef ANALYSE
        std::map<double, int> reward_list;
#endif
        bool _select = false;
        bool _uncover = false;
    };
    size_t count = 0;
    // type for mapping actions to values
    using qaction_t = std::map<size_t, qvalue_t>;

    // type for states
    using qstate_t = std::pair<std::vector<double>, std::vector<double>>;

    // type for mapping states to action-values
    using qtable_t = std::map<qstate_t, qaction_t>;

    // actual values
    qtable_t _Q;
public:
    // whether we are doing minimization or maximization
    bool _is_minimization = true;
    size_t _d_size = 0; // discrete state-vector size
    size_t _c_size = 0; // continuous state-vector size
    bool learning = true;
    
    //uncovered states
    //qtable_t uncovered;

public:

    /**
     * Converts a raw observation into appropriate format for Q-table
     * @param d_vars
     * @param c_vars
     * @return
     */
    qstate_t make_state(double* d_vars, double* c_vars) {
        std::vector<double> d_vector;
        std::vector<double> c_vector;
        if (d_vars != nullptr) {
            d_vector.resize(_d_size); // make space in vector
            for (size_t d = 0; d < _d_size; ++d) // copy over data
                d_vector[d] = d_vars[d];
        }
        if (c_vars != nullptr) {
            c_vector.resize(_c_size); // make space in vector
            for (size_t c = 0; c < _c_size; ++c) // copy over data
            {
                // truncates to "lump" several concrete states together to avoid a Q-table explosion
                c_vector[c] = std::trunc(c_vars[c]);
            }
        }
        return {d_vector, c_vector};
    }

    /**
     * Returns best known Q-value for the given state (over all actions)
     * @param d_vars
     * @param c_vars
     * @return
     */
    qvalue_t best_value(double* d_vars, double* c_vars) {
        auto state = make_state(d_vars, c_vars);

        // lets try to find a matching state
        auto it = _Q.find(state);
        qvalue_t best = {0, 0};
        if (it != _Q.end()) {
            auto& state_table = it->second;
            for (auto& other : state_table) {
                if (other.second._count == 0) continue;
                if (best._count == 0)
                    best = other.second;
                if (_is_minimization && other.second._value < best._value)
                    best = other.second;
                else if (!_is_minimization && other.second._value > best._value)
                    best = other.second;
            }
        }
        return best;
    }

public:

    QLearner(bool is_minimization, size_t d_size, size_t c_size) : _is_minimization(is_minimization), _d_size(d_size), _c_size(c_size) {
#ifdef VERBOSE
        std::cerr << "[New Q-Learner (" << this << ") with sizes (" << d_size << ", " << c_size << ") for minimization?=" << std::boolalpha << is_minimization << "]" << std::endl;
#endif
    }

    // this object is default copyable
    QLearner(const QLearner& other) = default;

    /**
     * Add an observation/sample. This modifies the Q-values of the given
     * state-vector pair. The value given is assumed to be the delta of the
     * observed cost/reward between the (d_vars,c_vars) pair to the (t_d_vars,t_c_vars)
     * pair when the action was used.
     * Notice that t_d_vars and t_c_vars may be null if the terminal state was
     * reached (i.e. a unique sink-state with a permanent q-value of zero).
     * @param d_vars discrete values of current state
     * @param c_vars continuous values of current state
     * @param action action used
     * @param t_d_vars discrete values of next state
     * @param t_c_vars continuous values of next state
     * @param value is the immediate reward/cost
     */

    void add_sample(double* d_vars, double* c_vars, size_t action, double* t_d_vars, double* t_c_vars, double v_reward) {
        const double gamma = 0.99; // discount, we could make it converge to zero by making this dependent on the number of samples seen for this state-action-pair
        const double alpha = 2.0; // constant learning rate
        double reward = v_reward;
        auto from_state = make_state(d_vars, c_vars);
        auto future_estimate = best_value(t_d_vars, t_c_vars);
        qvalue_t& q = _Q[from_state][action];
        const double learning_rate = 1.0 / std::min<double>(alpha, q._count + 1);
        //const double learning_rate = 1.0/alpha;
        assert(learning_rate <= 1.0);
        assert(future_estimate._value == 0 || future_estimate._count != 0);
        if (q._count == 0) {
            // special case, we have no old value            
            q._value = reward + gamma * future_estimate._value;
        } else {
            // standard Q-value update
            q._value = q._value + (learning_rate * (reward + (gamma * future_estimate._value) - q._value));
            //conservative Q-value
            /*if(q.min_reward > v_reward)
            {
                q.min_reward = v_reward;
            }
            reward = q.min_reward;
            q._value = q._value + reward + (gamma * future_estimate._value);*/
        }
        q._count += 1;
    }
    
     /**
     * Add a state-action pair uncovered by learning.
     * @param d_vars discrete values of current state
     * @param c_vars continuous values of current state
     * @param action action used
     */
    void add_uncovered(double* d_vars, double* c_vars, size_t action) {
        auto from_state = make_state(d_vars, c_vars);
        qvalue_t& q = _Q[from_state][action];
        q._count = 1;
        q._select = false;
        q._value = min_reward;
        q._uncover = true;
    }

    /**
     * Returns the statistics of the "mapped state", namely the range of the q-values 
     * (first constituents of return) and the total sum of samples seen (last 
     * constituent).
     * @param d_vars
     * @param c_vars
     * @return (lower,upper,sum_samples)
     */
    std::tuple<double, double, size_t, size_t> search_statistics(double* d_vars, double* c_vars) {
        auto state = make_state(d_vars, c_vars);
        auto it = _Q.find(state);
        double lower = std::numeric_limits<double>::infinity();
        double upper = -std::numeric_limits<double>::infinity();
        size_t sum_count = 0;
        size_t n_actions = 0;
        if (it != _Q.end()) {
            auto& state_table = it->second;
            for (auto& stats : state_table) {
                if (stats.second._count != 0) {
                    sum_count += stats.second._count;
                    lower = std::min(lower, stats.second._value);
                    upper = std::max(upper, stats.second._value);
                    ++n_actions;
                }
            }
        }
        return {lower, upper, sum_count, n_actions};
    }

    /**
     * Returns the Q-value for a given action (a) or the lowest (resp highest if maximization)
     * value of any other action (a') observed if no observation has yet been made
     * of the action (a).
     * @param d_vars
     * @param c_vars
     * @param action
     * @return
     */
    qvalue_t value(double* d_vars, double* c_vars, size_t action) {
        auto state = make_state(d_vars, c_vars);

        // lets try to find a matching state
        auto it = _Q.find(state);
        if (it != _Q.end()) {
            // we have observed this state before
            auto& state_table = it->second;
            auto action_it = state_table.find(action);
            if (action_it != state_table.end()) {
                // we have observations for this action, return the computed Q-value
                return action_it->second;
            } else {
                // No prior observation of the action, we need a default value.
                return {0, 0};
                //                if (_is_minimization)
                //                    return {min, 0};
                //                else
                //                    return {max, 0};
            }
        } else {
            // No prior observation of the state, we need a default value.
            return {0, 0};
            //            if (_is_minimization)
            //                return {min, 0};
            //            else
            //                return {max, 0};
        }
    }

    /**
     * Inspects whether the given action is the "best" for the given state.
     * I.e. if we minimize, it will be the action with the lowest Q-value.
     * Several actions can be equally good.
     * @param d_vars
     * @param c_vars
     * @param action
     * @return
     */
    bool is_allowed(double* d_vars, double* c_vars, size_t action, bool* found) {
        *found = true;
        qvalue_t current_v = value(d_vars, c_vars, action);
        qvalue_t best_v = best_value(d_vars, c_vars);

        assert(current_v._count == 0 || best_v._count != 0);
        
        if(current_v._uncover) {
            return false;
        }
        else {
            if (current_v._count > 0 && current_v._value == best_v._value) {
                return true;
            } else if (best_v._count == 0) {
                // if the current state and action is not found, 
                // then the action is allowed for exploration
                // return true;
                // if the current state and action is not found,
                // then the action is not allowed
                *found = false;
            }
        }

        return false;
    }

    int length() {
        if (&_Q != nullptr) return _Q.size();
        else return 0;
    }

    size_t d_size() {
        return _d_size;
    }
    
    void clear_strategy() {
        _Q.clear();
    }
    
    void print_complete_score_table(std::ostream& out) {
        bool first = true;
        out << "{\n";
        for (auto& state_action : _Q) {
            auto& state = state_action.first;
            auto& action_map = state_action.second;

            if (!first) out << ",\n"; // make json-friendly
            first = false;
            out << "\"(";
            // iterate over discrete state values
            for (auto& d_value : state.first) {
                out << d_value << ",";
            }
            out << "),[";
            // iterate over concrete/continuous state values
            for (auto& c_value : state.second) {
                out << c_value << ",";
            }
            out << "]\":{";
            bool first_action = true;
            for (auto& action_value : action_map) {
                if (!first_action) out << ",";
                first_action = false;
                out << "\n\t";
                //action_value.first is the action ID.
                //action_value.second is the value of the state-action pair, and the count of it.
                out << "\"" << action_value.first << "\":" << action_value.second._value;
            }
            out << "}";
        }
        out << "\n}";
    }

    void print_partial_score_table(std::ostream& out, bool compact, bool uncovered) {
        bool first = true;
        bool tag = false;
        
        out << "{\n";
        for (auto& state_action : _Q) {
            tag = false;
            auto& state = state_action.first;
            auto& action_map = state_action.second;
            for (auto& action_value : action_map) {
                if (compact && action_value.second._select) {
                    tag = true;
                }
                if (uncovered && action_value.second._uncover) {
                    tag = true;
                }
            }
            if (tag) {
                if (!first) out << ",\n"; // make json-friendly
                first = false;
                out << "\"(";
                // iterate over discrete state values
                for (auto& d_value : state.first) {
                    out << d_value << ",";
                }
                out << "),[";
                // iterate over concrete/continuous state values
                for (auto& c_value : state.second) {
                    out << c_value << ",";
                }
                out << "]\":{";
                bool first_action = true;
                for (auto& action_value : action_map) {
                    if (compact && action_value.second._select) {
                        if (!first_action) out << ",";
                        first_action = false;
                        out << "\n\t";
                        //action_value.first is the action ID.
                        //action_value.second is the value of the state-action pair, and the count of it.
                        out << "\"" << action_value.first << "\":" << action_value.second._value;
                    }
                    if (uncovered && action_value.second._uncover) {
                        if (!first_action) out << ",";
                        first_action = false;
                        out << "\n\t";
                        out << "\"" << action_value.first << "\":" << action_value.second._value;
                    }
                }
                out << "}";
            }
        }
        out << "\n}";
    }
        
    /**
     * Outputs the learned q-values to the string-stream in a json-friendly format
     * of a map over "(discrete,continuous)"-state variable vector pairs and
     * into maps from actions to q-values.
     * @param out - the output stream to write to, defaults to stderror.
     */
    void print(std::ostream& out = std::cerr) {
        if (learning) {
            learning = false;
            this->print_complete_score_table(out);
        } else {
            #if defined(COMPACT) && !defined(CEG) 
                this->print_partial_score_table(out, true, false);
            #endif
            #if !defined(COMPACT) && !defined(CEG) 
                this->print_partial_score_table(out, true, false);
            #endif
            
            #if defined(CEG) 
                this->print_partial_score_table(out, false, true);
            #endif
        }
    }

    void mark(double* d_vars, double* c_vars, size_t action) {
        //std::ostream& out = std::cerr;
        bool found = false;
        if (is_allowed(d_vars, c_vars, action, &found)) {
            auto state = make_state(d_vars, c_vars);
            auto it = _Q.find(state);
            if (it != _Q.end()) {
                auto& state_table = it->second;
                auto action_it = state_table.find(action);

                if (action_it != state_table.end()) {
                    action_it->second._select = true;
                } else {
                }
            }

        } else {
        }
    }
};


#endif /* EXTERNAL_LEARNING_H */

