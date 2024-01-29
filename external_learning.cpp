#include "external_learning.h"

// we use this to check that we do not deallocate an object twice.
// this should never happen, so this is *only* useful if you suspect that
// Uppaal Stratego is doing something wrong.
std::set<QLearner*> live;

/**
 * Allocates an instance of a learner
 * @param minimization, flag for determining optimization type (minimization=true/maximization=false)
 * @param d_size, size of the discrete array
 * @param c_size, size of the continuous array
 * @param a_size, number of (controllable) actions available in the system
 * @return a pointer to a learner object
 */
extern "C" void* uppaal_external_learner_alloc(bool minimization, size_t d_size, size_t c_size, size_t a_size) {
    auto object = new QLearner(minimization, d_size, c_size);
    live.insert(object); // for later sanitycheck
    std::cerr << "-----------------------------------------------------------\n";
    std::cerr << "External Q learning - v20240129:";
#ifdef NEAREST_NEIGHBOR
    std::cerr << " NN (Nearest Neighbor) Version.";
#endif
#ifdef COMPACT
    std::cerr << " Compact Strategy Version.";
#endif
#ifdef CEG
    std::cerr << " CEG Version.";
#endif
    std::cerr << "\n";
    return object;
}

int count = 0;

/**
 * Deallocation code for objects allocated by uppaal_external_learner_alloc
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 */
extern "C" void uppaal_external_learner_dealloc(void* object) {
    QLearner* obj = (QLearner*) object;
#ifndef ANALYSE
    std::cerr << "Learn: ";
#endif
#ifdef ANALYSE
    std::cerr << "Analyse: ";
#endif
    if (obj->_is_minimization) std::cerr << "min - ";
    else std::cerr << "max - ";
    std::cerr << count++ << ":: Q-table's length: " << obj->length() << "\n";
    //obj->reduce();
    if (obj != nullptr && live.count(obj) != 1) {
        assert(false && "Call-sequence from UPPAAL was wrong, please report to the UPPAAL developers");
    }

    delete obj;
    live.erase(obj);
    return;
}

/**
 * 
 * @param data
 * @param is_min
 * @param d_size
 * @param c_size
 * @param a_size
 * @return 
 */
extern "C" void* uppaal_external_learner_parse(const char* data, bool is_min, size_t d_size, size_t c_size, size_t a_size) {
    auto object = new QLearner(is_min, d_size, c_size);
    live.insert(object); // for later sanitycheck
    return object;
}

/**
 * Write the state of the learner (called by saveStrategy in uppaal)
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 */
extern "C" char* uppaal_external_learner_print(void* object) {
    std::stringstream outstream;
    QLearner* ql = (QLearner*) object;
#ifndef ANALYSE
    ql->print(outstream); // ask learning object to be printed to a stream
#endif
#ifdef ANALYSE
    ql->analyse_print(outstream);
#endif
    auto data = outstream.str(); // convert the stream into a regular string object
    char* tmp = new char[data.size() + 1]; // create a c-style string with enough space
    strcpy(tmp, data.c_str()); // copy over the data
    return tmp; // deallocation is handled by the caller
}

/**
 * Deep-copy function of an instance of a leaner.
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 * @return a pointer to a duplicate/deep-copy of object
 */
extern "C" void* uppaal_external_learner_clone(void* object) {
    assert(object != nullptr);
    auto new_object = new QLearner(*(QLearner*) object);
    live.insert(new_object);
    return new_object;
}

/**
 * Called for each sample in a trace. Given a trace on s_0-a->s_1-b-> .. s_n
 * samples a received in inverse-order (s_1-b->s_2 is seen before s_0-a->s_1)
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 * @param action, the action taken
 * @param from_d_vars, the discrete state-vector of the origin state
 * @param from_c_vars, the continuous state-vector of the origin state
 * @param t_d_vars, the discrete state-vector of the target state
 * @param t_c_vars, the continuous state-vector of the target state
 * @param value, the observed cost/reward (see @uppaal_external_learner_alloc, minimization)
 */
extern "C" void uppaal_external_learner_sample_handler(void* object, size_t action,
        double* from_d_vars, double* from_c_vars,
        double* t_d_vars, double* t_c_vars, double value) {
    if (object == nullptr) {
        return;
    }
    auto q = (QLearner*) object;
    //offline
    auto from_state = q->make_state(from_d_vars, from_c_vars);
    q->add_sample(from_d_vars, from_c_vars, action, t_d_vars, t_c_vars, value);
    return;
}

extern "C" void uppaal_external_learner_online_sample_handler(void* object, size_t action,
        double* from_d_vars, double* from_c_vars,
        double* t_d_vars, double* t_c_vars, double value) {
    return;
}

/**
 * Function for returning the result of the leaner; used both during training (is_eval=false)
 * and evaluation (is_eval=true)
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 * @param is_eval, indicating whether we are evaluating or training
 * @param action, indicating the action taken from the state where d_vars and t_vars were observed
 * @param d_vars, the observed discrete state-vector
 * @param t_vars, the observed continuous state-vector
 */
extern "C" double uppaal_external_learner_predict(void* object, bool is_eval, size_t action, double* d_vars, double* c_vars) {
    // you can control search here!
    // return ONLY weights > 0, non inf and non nan.
    // a weighted choice will be done over all actions according to the weight
    auto q = (QLearner*) object;
    double reward = 0.0;
    //    std::ostream& out = std::cerr;
    auto from_state = q->make_state(d_vars, c_vars);
    //    size_t to_action = action;
    bool found = false, allowed = false;

    if (!q->learning) {
        q->mark(d_vars, c_vars, action);
    }

    if (is_eval) {
        allowed = q->is_allowed(d_vars, c_vars, action, &found);
        if (allowed & found) {
            reward = 1.0;
        } else if(found) {
            reward = 0.0;
        } else {
            //Q-table does not contain the state
            /*
             * Somehow the function should inform the model checker
             * that the state is not found in the strategy so that
             * the verification stops as if meeting a deadlock
            */
            if (!q->learning) {    
                if(std::find(q->uncovered.begin(), q->uncovered.end(), from_state) == q->uncovered.end()){
                    q->uncovered.push_back(from_state);
                    std::cerr << "State <" << from_state.first[0] << "," << from_state.first[1] << "> is not found! \n";
                }
                //assert(false);
            }
            reward = 0.0;
        }
    } else {
        auto [lower, upper, sum_count, nactions] = q->search_statistics(d_vars, c_vars);
        auto value = q->value(d_vars, c_vars, action);

        if (sum_count == 0) {
            assert(value._count == 0);
            reward = 0;
        } else {
            const double pr_action = ((double) sum_count / (double) nactions);
            const double difference = upper - lower;

            if (difference == 0)
                return 1.0;

            // handle special-case where we want "best"-value when no samples are seen.
            double relative = value._count != 0 ? value._value :
                    (q->_is_minimization ? lower : upper); // if no samples, pick best value

            // compute normalization (between [0,1])
            if (q->_is_minimization)
                relative = (upper - relative) / difference;
            else
                relative = (relative - lower) / difference;

            // punish "more sampled" more; i.e. they will be even further from weight 1
            // the more samples they have seen
            const double lifted = std::pow(
                    relative, std::min(1000.0,
                    std::sqrt(std::max<double>(value._count, pr_action))));

            // r denotes the proportion of samples used for this given action
            // out of all samples passing through the state
            double r = 1.0;
            if (value._count > 0)
                r = std::sqrt(std::log(sum_count) / (double) value._count);

            // exploration fraction
            double C = 1.0 / nactions;

            // combine expressions, the "goodness" and the exploration-term.
            reward = lifted + (r * C) / (1.0 + C);
        }
    }
    //predict
    return reward;
}

/**
 * Batch-completion call-back
 * @param object, A pointer returned by @uppaal_external_learner_alloc
 */
extern "C" void uppaal_external_learner_flush(void* object) {
    // not used by q-learning
    return;
}