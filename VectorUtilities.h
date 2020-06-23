#ifndef VectorUtilities_H
#define VectorUtilities_H

#include <algorithm>
#include <functional>
#include <vector>
#include <assert.h>     /* assert */
#include <iterator>     // std::reverse_iterator


// Muchas Gracias https://stackoverflow.com/questions/3376124/how-to-add-element-by-element-of-two-stl-vectors

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(), 
                   std::back_inserter(result), std::plus<T>());
    return result;
}

#endif