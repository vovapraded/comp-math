from itertools import permutations
from math import log10, ceil
import numpy as np
from typing import Tuple


class GaussSeidelSolver:
    def __init__(self, max_iterations: int = 1000000):
        self.max_iterations = max_iterations
        self.tolerance = None
        self.matrix_A = None

    def is_diagonally_dominant(self, A: np.ndarray) -> bool:
        n = len(A)
        for i in range(n):
            sum_abs = sum(abs(A[i, j]) for j in range(n) if j != i)
            if abs(A[i, i]) <= sum_abs :
                return False
        return True

    def check_zeros_on_diagonal(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Проверяет наличие нулей на главной диагонали и переставляет строки, если такие нули есть.
        """
        n = len(A)
        for i in range(n):
            if A[i, i] == 0:
                # Находим строку с ненулевым элементом в текущем столбце и меняем их местами
                for j in range( n):
                    if A[j, i] != 0:
                        # Переставляем строки
                        A[[i, j], :] = A[[j, i], :]
                        b[i], b[j] = b[j], b[i]
                        print(f"Переставлены строки {i + 1} и {j + 1} для устранения деления на ноль.")
                        break
                else:
                    # Если не удается найти строку с ненулевым элементом, выбрасываем ошибку
                    raise ValueError(f"Невозможно устранить деление на ноль в A[{i}, {i}] и в столбце.")
        return A, b

    def process_matrix(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Проверяем, есть ли нули на главной диагонали

        # Проверяем, является ли матрица диагонально доминирующей
        if not self.is_diagonally_dominant(A):
            # Если не является, пытаемся привести её к диагонально доминирующему виду
            A, b, success = self.make_diagonally_dominant(A, b)
            if not success:
                print("Не удалось привести матрицу к диагонально доминирующей форме. Сходимость не гарантируется.")
                choice = input("Продолжить? (да/нет): ").lower()
                if choice != 'да':
                    exit()  # Выход из программы, если пользователь не согласен продолжать
            else:
                print("Матрица преобразована для диагонального преобладания.")

        # Проверяем ещё раз на нули после диагонального преобладания
        A, b = self.check_zeros_on_diagonal(A, b)

        # Копируем матрицу A для дальнейших вычислений
        self.matrix_A = A.copy()
        return A, b

    def make_diagonally_dominant(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        n = len(A)
        # Индексы строк, которые будем переставлять
        for i in range(n):
            # Находим строку с максимальным элементом в столбце i (по абсолютной величине)
            max_row = i
            for j in range(n):
                if abs(A[j, i]) >= abs(A[max_row, i]):
                    max_row = j
            # Если максимальный элемент в столбце не на диагонали, переставляем строки
            if max_row != i:
                A[[i, max_row]] = A[[max_row, i]]
                b[i], b[max_row] = b[max_row], b[i]
        # После перестановок проверяем, стала ли матрица диагонально доминирующей
        if self.is_diagonally_dominant(A):
            return A, b, True
        else:
            return A, b, False

    def input_matrix(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        A = np.zeros((n, n))
        b = np.zeros(n)
        print("Введите элементы матрицы A построчно (через пробел):")
        for i in range(n):
            row = list(map(float, input(f"Строка {i + 1}: ").split()))
            if len(row) != n:
                raise ValueError("Неверное количество элементов в строке")
            A[i] = row
        print("Введите элементы вектора b (через пробел):")
        b = np.array(list(map(float, input().split())))
        if len(b) != n:
            raise ValueError("Размер вектора b не соответствует матрице")
        return A, b



    def solve(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        if self.tolerance is None:
            raise ValueError("Погрешность не задана.")
        n = len(A)
        x = np.zeros(n)
        iterations = 0
        while iterations < self.max_iterations:
            x_new = np.copy(x)
            for i in range(n):
                sum_left = np.dot(A[i, :i], x_new[:i])  # элементы до диагонали
                sum_right = np.dot(A[i, i + 1:], x[i + 1:])  # элементы после диагонали
                sum_ax = sum_left + sum_right
                x_new[i] = (b[i] - sum_ax) / A[i, i]

            error = np.abs(x_new - x)
            if np.max(error) < self.tolerance:
                break
            x = x_new
            iterations += 1
        return x, iterations, error

    def solve_from_input(self) -> None:
        try:
            self.tolerance = float(input("Введите допустимую погрешность: "))
            n = int(input("Введите размерность матрицы (n ≤ 20): "))
            A, b = self.input_matrix(n)
            A, b = self.process_matrix(A, b)
            solution, iterations, error = self.solve(A, b)
            self.print_results(solution, iterations, error)
        except ValueError as e:
            print(f"Ошибка: {e}")

    def solve_from_file(self, filename: str) -> None:
        try:
            with open(filename, 'r') as file:
                self.tolerance = float(file.readline().strip())
                n = int(file.readline().strip())
                A = np.array([list(map(float, file.readline().strip().split())) for _ in range(n)])
                b = np.array(list(map(float, file.readline().strip().split())))
            A, b = self.process_matrix(A, b)
            solution, iterations, error = self.solve(A, b)
            self.print_results(solution, iterations, error)
        except FileNotFoundError:
            print(f"Файл {filename} не найден.")
        except ValueError as e:
            print(f"Ошибка в формате файла: {e}")


    def print_results(self, solution: np.ndarray, iterations: int, error: np.ndarray) -> None:
        decimals = max(1, ceil(-log10(self.tolerance)))
        print("\nРешение системы:")
        for i, val in enumerate(solution):
            print(f"x[{i + 1}] = {val:.{decimals}f}")
        print(f"Количество итераций: {iterations}")
        print(f"Использованная погрешность: {self.tolerance:.{decimals}f}")
        print("Вектор погрешностей:")
        for i, err in enumerate(error):
            print(f"e[{i + 1}] = {err:.{decimals+1}f}")
        if self.matrix_A is not None:
            norm_2 = np.linalg.norm(self.matrix_A, ord=2)
            print(f"Евклидова норма матрицы A: {norm_2:.{decimals}f}")


if __name__ == "__main__":
    solver = GaussSeidelSolver()
    choice = input("Выберите способ ввода данных (1 - с клавиатуры, 2 - из файла): ")
    if choice == "1":
        solver.solve_from_input()
    elif choice == "2":
        filename = input("Введите имя файла: ")
        solver.solve_from_file(filename)
    else:
        print("Неверный выбор.")