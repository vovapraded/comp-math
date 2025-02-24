import numpy as np
from typing import List, Union


class GaussSeidelSolver:
    def __init__(self, max_iterations: int = 1000):
        """
        Инициализация класса для метода Гаусса-Зейделя.

        Параметры:
        - max_iterations: максимальное количество итераций
        """
        self.max_iterations = max_iterations
        self.tolerance = None  # Погрешность будет задаваться позже
        self.matrix_A = None  # Для хранения матрицы A

    def is_diagonally_dominant(self, A: np.ndarray) -> bool:
        """
        Проверка матрицы на строгое диагональное преобладание.

        Возвращает True, если матрица строго диагонально доминирует, иначе False.
        """
        n = len(A)
        for i in range(n):
            sum_abs = sum(abs(A[i, j]) for j in range(n) if j != i)
            if abs(A[i, i]) <= sum_abs:
                return False
        return True

    def solve_from_input(self) -> None:
        """
        Решение СЛАУ с вводом данных с клавиатуры, включая погрешность.
        """
        try:
            # Ввод погрешности
            tolerance = float(input("Введите допустимую погрешность (например, 1e-6): "))
            if tolerance <= 0:
                raise ValueError("Погрешность должна быть положительным числом")

            # Установка погрешности
            self.tolerance = tolerance

            n = int(input("Введите размерность матрицы (n ≤ 20): "))
            if n > 20 or n <= 0:
                raise ValueError("Размер матрицы должен быть в диапазоне 1–20")

            # Ввод матрицы A и вектора b
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

            # Проверка диагонального преобладания
            if not self.is_diagonally_dominant(A):
                print("Матрица не строго диагонально доминирующая. Сходимость не гарантируется.")
                choice = input("Продолжить? (да/нет): ").lower()
                if choice != 'да':
                    return

            # Сохранение матрицы A
            self.matrix_A = A.copy()

            # Решение
            solution, iterations, error = self.solve(A, b)
            self.print_results(solution, iterations, error)

        except ValueError as e:
            print(f"Ошибка: {e}")
        except Exception as e:
            print(f"Произошла ошибка: {e}")

    def solve_from_file(self, filename: str) -> None:
        """
        Решение СЛАУ с вводом данных из файла, включая погрешность.

        Формат файла: первая строка — погрешность, вторая строка — размер n,
        затем n строк матрицы A, затем одна строка вектора b.
        """
        try:
            with open(filename, 'r') as file:
                # Чтение погрешности
                tolerance = float(file.readline().strip())
                if tolerance <= 0:
                    raise ValueError("Погрешность должна быть положительным числом")

                self.tolerance = tolerance

                # Чтение размера
                n = int(file.readline().strip())
                if n > 20 or n <= 0:
                    raise ValueError("Размер матрицы должен быть в диапазоне 1–20")

                A = np.zeros((n, n))
                for i in range(n):
                    row = list(map(float, file.readline().strip().split()))
                    if len(row) != n:
                        raise ValueError("Неверное количество элементов в строке матрицы")
                    A[i] = row

                b = np.array(list(map(float, file.readline().strip().split())))
                if len(b) != n:
                    raise ValueError("Размер вектора b не соответствует матрице")

            # Проверка диагонального преобладания
            if not self.is_diagonally_dominant(A):
                print("Матрица не строго диагонально доминирующая. Сходимость не гарантируется.")
                choice = input("Продолжить? (да/нет): ").lower()
                if choice != 'да':
                    return

            # Сохранение матрицы A
            self.matrix_A = A.copy()

            # Решение
            solution, iterations, error = self.solve(A, b)
            self.print_results(solution, iterations, error)

        except FileNotFoundError:
            print(f"Файл {filename} не найден.")
        except ValueError as e:
            print(f"Ошибка в формате файла: {e}")
        except Exception as e:
            print(f"Произошла ошибка: {e}")

    def solve(self, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Реализация метода Гаусса-Зейделя.

        Возвращает решение, количество итераций и вектор погрешностей.
        """
        if self.tolerance is None:
            raise ValueError("Погрешность не задана. Установите tolerance перед решением.")

        n = len(A)
        x = np.zeros(n)  # Начальное приближение
        iterations = 0

        while iterations < self.max_iterations:
            x_new = np.copy(x)
            for i in range(n):
                sum_ax = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - sum_ax) / A[i, i]

            # Проверка сходимости
            error = np.abs(x_new - x)
            if np.max(error) < self.tolerance:
                break

            x = x_new
            iterations += 1

        return x, iterations, error

    def print_results(self, solution: np.ndarray, iterations: int, error: np.ndarray) -> None:
        """
        Вывод результатов решения, включая евклидову норму матрицы.
        """
        print("\nРешение системы:")
        for i, val in enumerate(solution):
            print(f"x[{i + 1}] = {val:.6f}")
        print(f"Количество итераций: {iterations}")
        print(f"Использованная погрешность: {self.tolerance:.6f}")
        print("Вектор погрешностей (|x_i^(k) - x_i^(k-1)|):")
        for i, err in enumerate(error):
            print(f"e[{i + 1}] = {err:.6f}")

        # Вычисление и вывод евклидовой нормы матрицы A
        if self.matrix_A is not None:
            norm_2 = np.linalg.norm(self.matrix_A, ord=2)
            print(f"Евклидова норма матрицы A (2-норма): {norm_2:.6f}")


# Пример использования
if __name__ == "__main__":
    solver = GaussSeidelSolver()

    print("Выберите способ ввода данных:")
    print("1. С клавиатуры")
    print("2. Из файла")

    choice = input("Введите номер варианта (1 или 2): ")

    if choice == "1":
        solver.solve_from_input()
    elif choice == "2":
        filename = input("Введите имя файла: ")
        solver.solve_from_file(filename)
    else:
        print("Неверный выбор.")