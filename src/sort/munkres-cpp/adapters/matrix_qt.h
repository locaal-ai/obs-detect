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

#if !defined(_MUNKRES_ADAPTERS_QT_MATRIX_H_)
#define _MUNKRES_ADAPTERS_QT_MATRIX_H_

#include "munkres-cpp/matrix_base.h"
#include <QGenericMatrix>
#include <stdexcept>

namespace munkres_cpp {

template<class T, int N, int M>
class matrix_qt : public matrix_base<T>, public QGenericMatrix<N, M, T> {
public:
	matrix_qt(size_t, size_t) {}

	matrix_qt(const QGenericMatrix<N, M, T> &other)
		: QGenericMatrix<N, M, T>::QGenericMatrix(other)
	{
	}

	matrix_qt<T, N, M> &operator=(const QGenericMatrix<N, M, T> &other)
	{
		QGenericMatrix<N, M, T>::operator=(other);
		return *this;
	}

	const T &operator()(size_t row, size_t column) const override
	{
		return QGenericMatrix<N, M, T>::operator()(row, column);
	};

	T &operator()(size_t row, size_t column) override
	{
		return QGenericMatrix<N, M, T>::operator()(row, column);
	}

	size_t columns() const override { return M; }

	size_t rows() const override { return N; }

	void resize(size_t rows, size_t columns, T value = T(0)) override
	{
		(void)value;
		if (rows != this->rows() || columns != this->columns()) {
			throw std::logic_error(
				"Called function with inappropriate default implementation.");
		}
	}
};

} // namespace munkres_cpp

#endif /* !defined(_MUNKRES_ADAPTERS_QT_MATRIX_H_) */
