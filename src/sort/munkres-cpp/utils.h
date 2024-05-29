#if !defined(__MUNKRES_CPP_UTILS_H__)
#define __MUNKRES_CPP_UTILS_H__

#include <algorithm>
#include <cmath>

namespace munkres_cpp {

template<typename T, template<typename> class M>
typename std::enable_if<std::is_floating_point<T>::value>::type replace_infinites(M<T> &matrix)
{
	std::replace_if(
		matrix.begin(), matrix.end(), [](const T &v) { return std::isinf(v); },
		std::numeric_limits<T>::max());
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, bool>::type
is_data_invalid(const T &)
{
	return false;
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, bool>::type
is_data_invalid(const T &value)
{
	return std::signbit(value);
}

template<typename T>
typename std::enable_if<!std::is_integral<T>::value, bool>::type is_data_invalid(const T &value)
{
	return value < T(0) ||
	       !(std::fpclassify(value) == FP_ZERO || std::fpclassify(value) == FP_NORMAL);
}

template<typename T, template<typename> class M>
typename std::enable_if<std::is_unsigned<T>::value, bool>::type is_data_valid(const M<T> &)
{
	return true;
}

template<typename T, template<typename> class M>
typename std::enable_if<std::is_signed<T>::value, bool>::type is_data_valid(const M<T> &matrix)
{
	return !std::any_of(matrix.begin(), matrix.end(),
			    [](const T &v) { return is_data_invalid<T>(v); });
}

} // namespace munkres_cpp

#endif /* !defined(__MUNKRES_CPP_UTILS_H__) */
