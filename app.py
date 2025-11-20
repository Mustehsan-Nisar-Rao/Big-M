import streamlit as st
import numpy as np
import pandas as pd
import re

class SimplexSolver:
    def __init__(self):
        self.tableau = None
        self.basic_vars = []
        self.var_names = []
        self.num_decision_vars = 0
        self.num_constraints = 0
        self.M = 1000
        self.iterations = []
        self.needs_artificial = False
        
    def parse_coefficients(self, input_str, expected_count):
        """Parse coefficients from user input"""
        if 'x' in input_str.lower():
            input_str = input_str.replace(' ', '')
            coeffs = [0] * expected_count
            pattern = r'([+-]?\d*\.?\d*)x(\d+)'
            matches = re.findall(pattern, input_str.lower())
            
            for coeff_str, var_num in matches:
                if coeff_str == '' or coeff_str == '+':
                    coeff = 1
                elif coeff_str == '-':
                    coeff = -1
                else:
                    coeff = float(coeff_str)
                
                var_idx = int(var_num) - 1
                if 0 <= var_idx < expected_count:
                    coeffs[var_idx] = coeff
            
            return coeffs
        else:
            return list(map(float, input_str.split()))
    
    def build_tableau(self, obj_coeffs, constraints, constraint_types, rhs_values, is_maximization):
        self.is_maximization = is_maximization
        self.num_decision_vars = len(obj_coeffs)
        self.num_constraints = len(constraints)
        self.original_obj_coeffs = obj_coeffs.copy()
        
        # Convert minimization to maximization
        if not is_maximization:
            self.obj_coeffs = [-c for c in obj_coeffs]
        else:
            self.obj_coeffs = obj_coeffs.copy()
        
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        # Count additional variables
        for c_type in constraint_types:
            if c_type == '<=':
                slack_count += 1
            elif c_type == '>=':
                surplus_count += 1
                artificial_count += 1
            elif c_type == '=':
                artificial_count += 1
        
        self.needs_artificial = (artificial_count > 0)
        
        # Build variable names
        self.var_names = [f'x{i+1}' for i in range(self.num_decision_vars)]
        
        for i in range(slack_count):
            self.var_names.append(f'S{i+1}')
        
        for i in range(surplus_count):
            self.var_names.append(f's{i+1}')
        
        for i in range(artificial_count):
            self.var_names.append(f'A{i+1}')
        
        self.var_names.append('RHS')
        
        # Initialize tableau
        total_vars = self.num_decision_vars + slack_count + surplus_count + artificial_count
        self.tableau = np.zeros((self.num_constraints + 1, total_vars + 1))
        
        slack_idx = self.num_decision_vars
        surplus_idx = self.num_decision_vars + slack_count
        artificial_idx = self.num_decision_vars + slack_count + surplus_count
        
        s_counter = 0
        surplus_counter = 0
        a_counter = 0
        
        for i, (coeffs, c_type, rhs) in enumerate(zip(constraints, constraint_types, rhs_values)):
            self.tableau[i, :self.num_decision_vars] = coeffs
            
            if c_type == '<=':
                self.tableau[i, slack_idx + s_counter] = 1
                self.basic_vars.append(self.var_names[slack_idx + s_counter])
                s_counter += 1
            elif c_type == '>=':
                self.tableau[i, surplus_idx + surplus_counter] = -1
                self.tableau[i, artificial_idx + a_counter] = 1
                self.basic_vars.append(self.var_names[artificial_idx + a_counter])
                surplus_counter += 1
                a_counter += 1
            elif c_type == '=':
                self.tableau[i, artificial_idx + a_counter] = 1
                self.basic_vars.append(self.var_names[artificial_idx + a_counter])
                a_counter += 1
            
            self.tableau[i, -1] = rhs
        
        if self.needs_artificial:
            for i in range(self.num_decision_vars):
                self.tableau[-1, i] = -self.obj_coeffs[i]
            
            for i in range(artificial_count):
                self.tableau[-1, artificial_idx + i] = self.M
            
            for i, basic_var in enumerate(self.basic_vars):
                if basic_var.startswith('A'):
                    self.tableau[-1] -= self.M * self.tableau[i]
        else:
            self.tableau[-1, :self.num_decision_vars] = [-c for c in self.obj_coeffs]
    
    def save_iteration(self, iteration, pivot_info=None):
        """Save current tableau state"""
        iteration_data = {
            'iteration': iteration,
            'tableau': self.tableau.copy(),
            'basic_vars': self.basic_vars.copy(),
            'var_names': self.var_names.copy(),
            'pivot_info': pivot_info
        }
        self.iterations.append(iteration_data)
    
    def find_pivot_column(self):
        obj_row = self.tableau[-1, :-1]
        min_val = np.min(obj_row)
        
        if min_val >= -1e-6:
            return -1
        
        return np.argmin(obj_row)
    
    def find_pivot_row(self, pivot_col):
        ratios = []
        for i in range(self.num_constraints):
            if self.tableau[i, pivot_col] > 1e-10:
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))
        
        if all(r == float('inf') for r in ratios):
            return -1
        
        return np.argmin(ratios)
    
    def pivot(self, pivot_row, pivot_col):
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row] /= pivot_element
        
        for i in range(len(self.tableau)):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i] -= multiplier * self.tableau[pivot_row]
        
        self.basic_vars[pivot_row] = self.var_names[pivot_col]
    
    def solve(self):
        iteration = 0
        self.save_iteration(iteration)
        
        max_iterations = 100
        
        while iteration < max_iterations:
            iteration += 1
            
            pivot_col = self.find_pivot_column()
            
            if pivot_col == -1:
                artificial_in_basis = False
                for i, basic_var in enumerate(self.basic_vars):
                    if basic_var.startswith('A') and abs(self.tableau[i, -1]) > 1e-6:
                        artificial_in_basis = True
                        break
                
                if artificial_in_basis:
                    return "infeasible", "No feasible solution exists (artificial variables in basis)"
                
                return "optimal", self.get_solution()
            
            pivot_row = self.find_pivot_row(pivot_col)
            
            if pivot_row == -1:
                return "unbounded", f"Problem is unbounded in direction of {self.var_names[pivot_col]}"
            
            pivot_info = {
                'row': pivot_row,
                'col': pivot_col,
                'element': self.tableau[pivot_row, pivot_col],
                'entering': self.var_names[pivot_col],
                'leaving': self.basic_vars[pivot_row]
            }
            
            self.pivot(pivot_row, pivot_col)
            self.save_iteration(iteration, pivot_info)
        
        return "max_iterations", "Maximum iterations reached"
    
    def get_solution(self):
        solution = {}
        for i in range(self.num_decision_vars):
            var_name = f'x{i+1}'
            if var_name in self.basic_vars:
                idx = self.basic_vars.index(var_name)
                solution[var_name] = self.tableau[idx, -1]
            else:
                solution[var_name] = 0
        
        optimal_value = self.tableau[-1, -1]
        if not self.is_maximization:
            optimal_value = -optimal_value
        
        solution['Z'] = optimal_value
        return solution

def create_tableau_dataframe(iteration_data):
    """Create a pandas DataFrame from tableau data"""
    tableau = iteration_data['tableau']
    var_names = iteration_data['var_names']
    basic_vars = iteration_data['basic_vars']
    
    # Create column headers
    columns = [v for v in var_names]
    
    # Create row data
    rows = []
    row_labels = []
    
    for i in range(len(basic_vars)):
        rows.append(tableau[i])
        row_labels.append(basic_vars[i])
    
    # Add objective row
    rows.append(tableau[-1])
    row_labels.append('Z')
    
    df = pd.DataFrame(rows, columns=columns, index=row_labels)
    return df

def main():
    st.set_page_config(page_title="Simplex Method Solver", layout="wide", page_icon="📊")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2ca02c;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1f77b4;
            margin-bottom: 1rem;
        }
        .success-box {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #4caf50;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #ffebee;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #f44336;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #ff9800;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📊 Simplex Method Solver (Big M Method)</div>', unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.header("🔧 Problem Configuration")
        
        # Number of variables and constraints
        num_vars = st.number_input("Number of Decision Variables", min_value=2, max_value=10, value=2, step=1)
        num_constraints = st.number_input("Number of Constraints", min_value=1, max_value=10, value=2, step=1)
        
        # Objective function type
        obj_type = st.radio("Objective Function Type", ["Maximize", "Minimize"])
        is_maximization = (obj_type == "Maximize")
        
        st.markdown("---")
        st.subheader("🎯 Objective Function")
        st.caption("Enter coefficients for the objective function")
        
        obj_coeffs = []
        cols = st.columns(min(num_vars, 3))
        for i in range(num_vars):
            with cols[i % 3]:
                coeff = st.number_input(f"c{i+1} (for x{i+1})", value=0.0, step=0.1, format="%.2f", key=f"obj_{i}")
                obj_coeffs.append(coeff)
        
        st.markdown("---")
        st.subheader("📋 Constraints")
        
        constraints = []
        constraint_types = []
        rhs_values = []
        
        for i in range(num_constraints):
            st.caption(f"**Constraint {i+1}**")
            
            constraint_coeffs = []
            cols = st.columns(min(num_vars, 3))
            for j in range(num_vars):
                with cols[j % 3]:
                    coeff = st.number_input(f"a{i+1}{j+1}", value=0.0, step=0.1, format="%.2f", key=f"const_{i}_{j}")
                    constraint_coeffs.append(coeff)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                c_type = st.selectbox(f"Type", ["<=", ">=", "="], key=f"type_{i}")
            with col2:
                rhs = st.number_input(f"RHS", value=0.0, step=0.1, format="%.2f", key=f"rhs_{i}")
            
            # Ensure RHS is non-negative
            if rhs < 0:
                constraint_coeffs = [-c for c in constraint_coeffs]
                rhs = -rhs
                if c_type == '<=':
                    c_type = '>='
                elif c_type == '>=':
                    c_type = '<='
            
            constraints.append(constraint_coeffs)
            constraint_types.append(c_type)
            rhs_values.append(rhs)
            
            st.markdown("---")
        
        solve_button = st.button("🚀 Solve Problem", type="primary", use_container_width=True)
    
    # Main content area
    if solve_button:
        # Display problem formulation
        st.markdown('<div class="sub-header">📝 Problem Formulation</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            
            # Objective function
            obj_str = f"**{obj_type}:** Z = "
            terms = []
            for i, coeff in enumerate(obj_coeffs):
                if coeff != 0:
                    if coeff > 0 and terms:
                        terms.append(f"+ {coeff}x{i+1}")
                    elif coeff < 0:
                        terms.append(f"- {abs(coeff)}x{i+1}")
                    else:
                        terms.append(f"{coeff}x{i+1}")
            obj_str += " ".join(terms) if terms else "0"
            st.markdown(obj_str)
            
            # Constraints
            st.markdown("**Subject to:**")
            for i, (coeffs, c_type, rhs) in enumerate(zip(constraints, constraint_types, rhs_values)):
                terms = []
                for j, coeff in enumerate(coeffs):
                    if coeff != 0:
                        if coeff > 0 and terms:
                            terms.append(f"+ {coeff}x{j+1}")
                        elif coeff < 0:
                            terms.append(f"- {abs(coeff)}x{j+1}")
                        else:
                            terms.append(f"{coeff}x{j+1}")
                const_str = " ".join(terms) if terms else "0"
                st.markdown(f"- {const_str} {c_type} {rhs}")
            
            # Non-negativity
            non_neg = ", ".join([f"x{i+1}" for i in range(num_vars)])
            st.markdown(f"- {non_neg} ≥ 0")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Solve the problem
        solver = SimplexSolver()
        solver.build_tableau(obj_coeffs, constraints, constraint_types, rhs_values, is_maximization)
        
        # Show method information
        if solver.needs_artificial:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("⚠️ **Big M Method** is being used (artificial variables detected)")
            st.markdown(f"**M value:** {solver.M}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("ℹ️ **Standard Simplex Method** (no artificial variables needed)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        status, result = solver.solve()
        
        # Display iterations
        st.markdown('<div class="sub-header">🔄 Solution Iterations</div>', unsafe_allow_html=True)
        
        for iter_data in solver.iterations:
            iteration_num = iter_data['iteration']
            pivot_info = iter_data['pivot_info']
            
            with st.expander(f"**Iteration {iteration_num}**" + (" - Initial Tableau" if iteration_num == 0 else ""), expanded=(iteration_num <= 1)):
                if pivot_info:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Entering Variable", pivot_info['entering'])
                    with col2:
                        st.metric("Leaving Variable", pivot_info['leaving'])
                    with col3:
                        st.metric("Pivot Row", pivot_info['row'] + 1)
                    with col4:
                        st.metric("Pivot Column", pivot_info['col'] + 1)
                    
                    st.info(f"🎯 Pivot Element: **{pivot_info['element']:.4f}**")
                
                # Display tableau
                df = create_tableau_dataframe(iter_data)
                
                # Style the dataframe
                def highlight_pivot(s, pivot_info):
                    if pivot_info is None:
                        return [''] * len(s)
                    
                    styles = [''] * len(s)
                    if s.name == iter_data['basic_vars'][pivot_info['row']]:
                        styles[pivot_info['col']] = 'background-color: #ffeb3b; font-weight: bold'
                    return styles
                
                styled_df = df.style.format("{:.4f}").apply(highlight_pivot, pivot_info=pivot_info, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Show current objective value
                current_z = iter_data['tableau'][-1, -1]
                if not is_maximization:
                    current_z = -current_z
                st.caption(f"Current Objective Value: Z = {current_z:.4f}")
        
        # Display final result
        st.markdown('<div class="sub-header">🎉 Final Result</div>', unsafe_allow_html=True)
        
        if status == "optimal":
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ Optimal Solution Found!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Decision Variables:**")
                for i in range(num_vars):
                    var_name = f'x{i+1}'
                    value = result[var_name]
                    st.markdown(f"- **{var_name}** = {value:.4f}")
            
            with col2:
                st.markdown("**Objective Function Value:**")
                st.markdown(f"### **Z = {result['Z']:.4f}**")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights
            with st.expander("📊 Solution Insights"):
                st.markdown("**Basic Variables in Final Solution:**")
                for var in solver.basic_vars:
                    if not var.startswith('S') and not var.startswith('s') and not var.startswith('A'):
                        idx = solver.basic_vars.index(var)
                        value = solver.tableau[idx, -1]
                        st.markdown(f"- {var} = {value:.4f}")
                
                st.markdown("**Non-basic Variables (at zero):**")
                non_basic = [v for v in solver.var_names[:-1] if v not in solver.basic_vars and v.startswith('x')]
                if non_basic:
                    st.markdown(", ".join(non_basic))
                else:
                    st.markdown("None")
        
        elif status == "infeasible":
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.markdown("### ❌ No Feasible Solution")
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif status == "unbounded":
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("### ⚠️ Unbounded Solution")
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.markdown("### ⚠️ Solver Issue")
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show instructions when no problem is solved
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 👋 Welcome to the Simplex Method Solver!
        
        **Instructions:**
        1. Configure your optimization problem using the sidebar
        2. Set the number of decision variables and constraints
        3. Choose whether to maximize or minimize
        4. Enter the objective function coefficients
        5. Enter each constraint's coefficients, inequality type, and RHS value
        6. Click "🚀 Solve Problem" to see the solution
        
        **Features:**
        - ✅ Supports maximization and minimization problems
        - ✅ Handles ≤, ≥, and = constraints
        - ✅ Uses Big M Method for artificial variables
        - ✅ Shows detailed iteration-by-iteration tableaus
        - ✅ Highlights pivot elements and operations
        - ✅ Provides comprehensive solution insights
        
        **Note:** All decision variables are assumed to be non-negative (≥ 0)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show an example
        with st.expander("📚 Example Problem"):
            st.markdown("""
            **Maximize:** Z = 3x₁ + 2x₂
            
            **Subject to:**
            - 2x₁ + x₂ ≤ 18
            - 2x₁ + 3x₂ ≤ 42
            - 3x₁ + x₂ ≤ 24
            - x₁, x₂ ≥ 0
            
            **How to input:**
            1. Set 2 decision variables, 3 constraints
            2. Select "Maximize"
            3. Enter objective coefficients: c₁ = 3, c₂ = 2
            4. Enter constraint 1: 2, 1, ≤, 18
            5. Enter constraint 2: 2, 3, ≤, 42
            6. Enter constraint 3: 3, 1, ≤, 24
            7. Click Solve!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Simplex Method Solver v1.0</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
