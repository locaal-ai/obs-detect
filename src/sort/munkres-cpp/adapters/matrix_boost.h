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

#if !defined(_MUNKRES_ADAPTERS_BOOST_MATRIX_H_)
#define _MUNKRES_ADAPTERS_BOOST_MATRIX_H_

#include "munkres-cpp/matrix_base.h"
#include <boost/numeric/ublas/matrix.hpp>

namespace munkres_cpp {

template<class T>
class matrix_boost : public matrix_base<T>, public boost::numeric::ublas::matrix<T> {
public:
	matrix_boost(size_t rows, size_t columns)
		: boost::numeric::ublas::matrix<T>::matrix(rows, columns)
	{
		std::fill(matrix_base<T>::begin(), matrix_base<T>::end(), T(0));
	}

	matrix_boost(const boost::numeric::ublas::matrix<T> &other)
		: boost::numeric::ublas::matrix<T>::matrix(other)
	{
	}

	matrix_boost<T> &operator=(const boost::numeric::ublas::matrix<T> &other)
	{
		boost::numeric::ublas::matrix<T>::operator=(other);
		return *this;
	}

	const T &operator()(size_t row, size_t column) const override
	{
		return boost::numeric::ublas::matrix<T>::operator()(row, column);
	};

	T &operator()(size_t row, size_t column) override
	{
		return boost::numeric::ublas::matrix<T>::operator()(row, column);
	}

	size_t columns() const override { return boost::numeric::ublas::matrix<T>::size2(); }

	size_t rows() const override { return boost::numeric::ublas::matrix<T>::size1(); }

	void resize(size_t rows, size_t columns, T value = T(0)) override
	{
		if (rows != this->rows() || columns != this->columns()) {
			const auto rows_old = this->rows();
			const auto columns_old = this->columns();
			boost::numeric::ublas::matrix<T>::resize(rows, columns, true);
			if (rows_old < rows) {
				for (size_t i = rows_old; i < rows; ++i) {
					for (size_t j = 0; j < columns; ++j) {
						this->operator()(i, j) = value;
					}
				}
			}
			if (columns_old < columns) {
				for (size_t i = columns_old; i < columns; ++i) {
					for (size_t j = 0; j < rows; ++j) {
						this->operator()(j, i) = value;
					}
				}
			}
		}
	}
};

} // namespace munkres_cpp

#endif /* !defined(_MUNKRES_ADAPTERS_BOOST_MATRIX_H_) */
