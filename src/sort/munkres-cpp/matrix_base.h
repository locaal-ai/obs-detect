/*
 *   Copyright (c) 2016 Gluttton <gluttton@ukr.net>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#if !defined(__MUNKRES_CPP_MATRIX_BASE_H__)
#define __MUNKRES_CPP_MATRIX_BASE_H__

#include <iterator>

namespace munkres_cpp {

template<typename T> struct matrix_base {
	// Types.
	using value_type = T;

	// Interface.
	virtual const value_type &operator()(size_t, size_t) const = 0;
	virtual value_type &operator()(size_t, size_t) = 0;
	virtual size_t columns() const = 0;
	virtual size_t rows() const = 0;
	virtual void resize(size_t, size_t, value_type = value_type(0)) = 0;

	// Default implementation.
	virtual ~matrix_base() = default;

	// Allow to use standard algorithms.
	template<typename M> struct iterator {
		iterator(M &m, size_t r, size_t c) : m{m}, r{r}, c{c} {}
		typename std::conditional<std::is_const<M>::value, const typename M::value_type,
					  typename M::value_type>::type &
		operator*() const
		{
			return m(r, c);
		}
		bool operator==(const iterator &that)
		{
			return this->r == that.r && this->c == that.c;
		}
		bool operator!=(const iterator &that) { return !operator==(that); }
		iterator &operator++()
		{
			r += ++c / m.columns();
			c = c % m.columns();
			return *this;
		}

		M &m;
		size_t r, c;

		using difference_type = std::ptrdiff_t;
		using value_type = T;
		using pointer = T *;
		using reference = T &;
		using iterator_category = std::input_iterator_tag;
	};

	template<typename M> using const_iterator = iterator<const M>;

	iterator<matrix_base> begin() { return iterator<matrix_base>{*this, 0, 0}; }
	iterator<matrix_base> end() { return iterator<matrix_base>{*this, rows(), 0}; }
	const_iterator<matrix_base> begin() const
	{
		return const_iterator<matrix_base>{*this, 0, 0};
	}
	const_iterator<matrix_base> end() const
	{
		return const_iterator<matrix_base>{*this, rows(), 0};
	}
};

} // namespace munkres_cpp

#endif /* !defined(__MUNKRES_CPP_MATRIX_BASE_H__) */
