from itertools import permutations
from math import log10, ceil
import numpy as np
from typing import Tuple


class GaussSeidelSolver:
    def __init__(self, max_iterations: int = 1000):
        self.max_iterations = max_iterations
        self.tolerance = None
        self.matrix_A = None

    def is_diagonally_dominant(self, A: np.ndarray) -> bool:
        n = len(A)
        for i in range(n):
            sum_abs = sum(abs(A[i, j]) for j in range(n) if j != i)
            if abs(A[i, i]) <= sum_abs:
                return False
        return True

    def make_diagonally_dominant(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        n = len(A)
        indices = list(range(n))

        for perm in permutations(indices):
            permuted_A = A[list(perm), :]
            permuted_b = b[list(perm)]
            if self.is_diagonally_dominant(permuted_A):
                return permuted_A, permuted_b, True
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

    def process_matrix(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_diagonally_dominant(A):
            A, b, success = self.make_diagonally_dominant(A, b)
            if not success:
                print("Не удалось привести матрицу к диагонально доминирующей форме. Сходимость не гарантируется.")
                choice = input("Продолжить? (да/нет): ").lower()
                if choice != 'да':
                    exit()
            else:
                print("Матрица преобразована для диагонального преобладания.")
        self.matrix_A = A.copy()
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
                sum_ax = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
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