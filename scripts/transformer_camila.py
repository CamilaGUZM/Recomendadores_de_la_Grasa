import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DefineLimits(BaseEstimator, TransformerMixin):
    """
    Change columns that contain punctual, minimum and maximum values
    into two separate columns that indicate the limits of an observation,
    plus one that determines if an observation is of instance "No Gotea"

    
    Gets - dataframes with varying ranges of values within observations
         - COLUMNS: a list of columns to be modified, if any, applies to all dataframe

         - MARGINS:  which ads or subtracts the maximum found value by itself times the value of margin

         - VERBOSE: a boolean; when true, prints expansions made, limits determined per column and summary of data


    Out - dataframe that, per each of its original conflicting columns, 
          it will have two new columns with the maximum and minimum per column per observation, 

            - if any given column contains a value "No Gotea", it creates a new binary column
              accordingly which maps 1 in observations that contain it 
              (min = max = 0 for that observation) and 0 elsewhere.


    FIT: it computes minimum and maximum registered values

            - expands if needed
            - creates No Gotea column id needed

    TRANSFORM: 
            - adjust values to fit between memorized global minimums and maximums of each column
                - if expansion is needed, updates all expanded limits to the new ones
            - if memorized dataframe didn't create a No Gotea column, it won't for new data frames
        
    """

    def __init__(self, columns=None, margin=0.0, verbose=False):

        #Take attributes
        self.columns = columns
        self.margin = float(margin)
        self.verbose = verbose

        # Persistent states
        self.col_global_min_ = {}
        self.col_global_max_ = {}

        self.col_has_expanded_min_ = {}
        self.col_has_expanded_max_ = {}
        self.col_is_nogotea_ = {}

        # creates  empty list to store a history/log of all expansions 
        # that occur during both the fit() and transform() operations
        self.expansion_log = [] 
        

    # ----------------------------------------------------------------------
    #                       VECTORIZED PARSER HELPER
    # ----------------------------------------------------------------------
    @staticmethod
    def _parse_series(series):

        """  
        Helper gunction that converts columns observations to clean data, handles misproduced values
        from previous transformer pipelines

        Gets: pandas series that represents a whole column

        Out: 
            - types: categorizes values to know how to process them in later stages
                     - 'num' - Regular numeric value
                     - 'lt' - Less-than notation (<x)
                     - 'gt' - Greater-than notation (>x)
                     - 'range' - Range notation (x-y)
                     - 'nogotea' - "No Gotea" text
                     - 'invalid' - Unparseable value ('N/A', 'unknown', empty string)
            - numeric: raw values
            - range_mins: lower values of [x-y] type observations
            - range_maxs: upper values of [x-y] type observations
        """

        #transform recieved column values to string
        ser_str = series.astype(str)
        
        # Clean up string
        ser_str = ser_str.str.replace("＜", "<", regex=False) #map characters to conventional <,> not unicode
        ser_str = ser_str.str.replace("＞", ">", regex=False)
        ser_str = ser_str.str.replace(" ", "", regex=True)


        #----------------BOOLEANS----------------
        # no gotea
        lower = ser_str.str.lower()
        is_nogotea = lower == "nogotea"
        
        # dash boolean; Handle ranges like "265-295"
        has_dash = ser_str.str.contains(r'^\d+-\d+$', regex=True)
        
        #<,>  boolean mask per observation; Handle < and > values
        starts_lt = ser_str.str.startswith("<", na=False)   
        starts_gt = ser_str.str.startswith(">", na=False)
        


        # ---------------- ATTTEND RANGES ----------------------------
        # For ranges, create empty lists
        range_mins = np.full(len(ser_str), np.nan)
        range_maxs = np.full(len(ser_str), np.nan)
        
        if has_dash.any():
            # extract in ser_str only where has_dash is true;
            #returns only ranges values
            ranges = ser_str[has_dash]
            split_ranges = ranges.str.split("-", expand=True)
            range_mins[has_dash] = pd.to_numeric(split_ranges[0], errors='coerce')
            range_maxs[has_dash] = pd.to_numeric(split_ranges[1], errors='coerce')
        
        #------------------------------------------------------------------------
        


        # For regular numeric (removing < and >) and catches untreated dashes
        numeric_str = ser_str.str.replace(r"^[<>]", "", regex=True)
        numeric_str = numeric_str.str.replace(r"-.+$", "", regex=True)
        numeric = pd.to_numeric(numeric_str, errors="coerce")
        

        # ---------------- ASSIGN TYPES ----------------------------------------
        # create list with nums
        types = np.array(["num"] * len(ser_str), dtype=object)


        types[is_nogotea.values] = "nogotea"
        types[starts_lt.values] = "lt"
        types[starts_gt.values] = "gt"
        types[has_dash.values] = "range"
        
        # attend unassigned values, numerics stay of type "num"
        invalid_mask = (types == "num") & (numeric.isna().values)
        types[invalid_mask] = "invalid"
        #------------------------------------------------------------------------
        
        return types, numeric, range_mins, range_maxs


    # ----------------------------------------------------------------------
    #                       LOG EXPANSION HELPER
    # ----------------------------------------------------------------------

    # FINISH 

    def _log_expansion(self, stage, col, limit_type, old_value, new_value, reason=""):
        """
        Function that saves the historical changes


        """
        expansion_info = {
            "stage": stage,
            "column": col,
            "limit_type": limit_type,
            "old_value": old_value,
            "new_value": new_value,
            "margin": self.margin,
            "reason": reason
        }
        self.expansion_log.append(expansion_info)
        
        if self.verbose:
            margin_str = f" (margin={self.margin})" if self.margin != 0 else ""
            print(f"[{stage.upper()}] {col}: {limit_type.upper()} EXPANDED {old_value} → {new_value}{margin_str} {reason}")

    # ----------------------------------------------------------------------
    #
    #                                FIT
    #
    # ----------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        """
        X = X.copy()
        # if list of columns, use; if None, do all datafram
        cols = self.columns or list(X.columns)
        
        #------------------Print which Stage------------------
        if self.verbose:
            print("=" * 80)
            print("FITTING STAGE")
            print("=" * 80)
        #-----------------------------------------------------


        for col in cols:
            #extract series
            ser = X[col]
            #parse info
            types, nums, range_mins, range_maxs = self._parse_series(ser)

            # No Gotea Boolean; save in dictionary per column for transform
            has_nogotea = (types == "nogotea").any()
            self.col_is_nogotea_[col] = bool(has_nogotea)

            #--------------------- STORE VALID NUMBERS------------------------------------------
            #initialize valid numeric values (including from ranges)
            valid_nums = []
            
            # Regular numbers
            num_mask = (types == "num") & np.isfinite(nums)
            valid_nums.extend(nums[num_mask].tolist())
            
            # <, > BOOLEANS, values (use the numeric part)
            lt_mask = (types == "lt") & np.isfinite(nums)
            gt_mask = (types == "gt") & np.isfinite(nums)


            # For <x, the actual value is less than x, so x is an UPPER bound
            # For >x, the actual value is greater than x, so x is a LOWER bound
            valid_nums.extend(nums[lt_mask].tolist())
            valid_nums.extend(nums[gt_mask].tolist())
            
            # Range values BOOLEAN
            range_mask = (types == "range")
            valid_nums.extend(range_mins[range_mask & np.isfinite(range_mins)].tolist())
            valid_nums.extend(range_maxs[range_mask & np.isfinite(range_maxs)].tolist())
            # ----------------------------------------------------------------------------------



            if len(valid_nums) == 0:
                init_min = 0.0
                init_max = 0.0
            else:
                init_min = float(min(valid_nums))
                init_max = float(max(valid_nums))

            global_min = init_min
            global_max = init_max
            expanded_min = False
            expanded_max = False
            expansion_reason_min = ""
            expansion_reason_max = ""
            
            
            #--------------------------------  UPPER BOUNDS  ---------------------------------------------------
            #  x is an UPPER bound, and  might need to expand LOWER bound downward
            lt_mask = (types == "lt") & np.isfinite(nums)
            if lt_mask.any():
                vals_lt = nums[lt_mask]
                if len(vals_lt) > 0:
                    # consider expanding the LOWER bound (global_min) downward
                    # by looking at the smallest < value
                    min_lt_value = float(vals_lt.min())
                    
                    # If margin > 0, expand downward from the smallest < value
                    if self.margin > 0:
                        old = global_min
                        #if there is non zero expansion, else, expand by 0
                        expansion_amount = self.margin * abs(min_lt_value) if min_lt_value != 0 else self.margin
                        global_min = min_lt_value - expansion_amount
                        expanded_min = True

                        #save all < values in print
                        expansion_reason_min = f"due to < values (actual < {sorted(set(vals_lt))})"
                        self._log_expansion("fit", col, "min", old, global_min, expansion_reason_min)
                    
                    #------------------Print No Expansion due to 0 margin------------------
                    if self.verbose and self.margin == 0:
                        print(f"[FIT] {col}: Found < values {sorted(set(vals_lt))} but margin=0, so no expansion")

            #--------------------------------  UPPER BOUNDS  ---------------------------------------------------
            # x is a LOWER bound, might need to expand UPPER bound upward
            gt_mask = (types == "gt") & np.isfinite(nums)
            if gt_mask.any():
                vals_gt = nums[gt_mask]
                if len(vals_gt) > 0:
                    # expanding the UPPER bound (global_max) upward
                    # by looking at the largest > value
                    max_gt_value = float(vals_gt.max())
                    
                    # If margin > 0, expand upward from the largest > value
                    if self.margin > 0:
                        old = global_max
                        #if there is non zero expansion, else, expand by 0
                        expansion_amount = self.margin * abs(max_gt_value) if max_gt_value != 0 else self.margin
                        global_max = max_gt_value + expansion_amount
                        expanded_max = True

                        #save all > values in print
                        expansion_reason_max = f"due to > values (actual > {sorted(set(vals_gt))})"
                        self._log_expansion("fit", col, "max", old, global_max, expansion_reason_max)

                    #------------------Print No Expansion due to 0 margin------------------
                    if self.verbose and self.margin == 0:
                        print(f"[FIT] {col}: Found > values {sorted(set(vals_gt))} but margin=0, so no expansion")

            # Update attributes of column
            self.col_global_min_[col] = float(global_min)
            self.col_global_max_[col] = float(global_max)
            self.col_has_expanded_min_[col] = bool(expanded_min)
            self.col_has_expanded_max_[col] = bool(expanded_max)
            
            #------------------Print FINAL limits and types counts------------------
            if self.verbose:
                print(f"[FIT] {col}: Final limits = [{global_min}, {global_max}]")
                type_counts = pd.Series(types).value_counts()
                if len(type_counts) > 0:
                    print(f"[FIT] {col}: Type counts: {type_counts.to_dict()}")

        return self

    # ----------------------------------------------------------------------
    #
    #                              TRANSFORM  
    #  
    # ----------------------------------------------------------------------
    def transform(self, X):
        """
        """
        X = X.copy()
        cols = self.columns or list(X.columns)

        #------------------Print which Stage------------------
        if self.verbose:
            print("\n" + "=" * 80)
            print("TRANSFORMING STAGE")
            print("=" * 80)
        #------------------------------------------------------

        # initialize dictionaries
        out = {}
        pending_new_min = {}
        pending_new_max = {}

        for col in cols:
            ser = X[col]

            #get parse info
            types, nums, range_mins, range_maxs = self._parse_series(ser)

            #fetch fit limits
            global_min = self.col_global_min_[col]
            global_max = self.col_global_max_[col]

            #get booleans
            expanded_min_in_fit = self.col_has_expanded_min_[col]
            expanded_max_in_fit = self.col_has_expanded_max_[col]
            is_nogotea_col = self.col_is_nogotea_.get(col, False)

            # initialize list for columns
            n = len(ser)
            col_min = np.full(n, np.nan, dtype=float)
            col_max = np.full(n, np.nan, dtype=float)
            col_NG = np.zeros(n, dtype=int)

            # ---------------- No Gotea ----------------
            mask_ng = (types == "nogotea")
            if is_nogotea_col:
                col_min[mask_ng] = 0.0
                col_max[mask_ng] = 0.0
                col_NG[mask_ng] = 1

            # ---------------- Punctual numbers ----------------
            mask_num = (types == "num") & np.isfinite(nums)
            col_min[mask_num] = nums[mask_num]
            col_max[mask_num] = nums[mask_num]

            # ---------------- Ranges ----------------
            mask_range = (types == "range")
            col_min[mask_range] = range_mins[mask_range]
            col_max[mask_range] = range_maxs[mask_range]

            # ---------------- <x values  ----------------
            mask_lt = (types == "lt") & np.isfinite(nums)
            if mask_lt.any():
                # For <x values: min = global_min, max = global_max
                col_min[mask_lt] = global_min
                col_max[mask_lt] = global_max
                
                # Check if  expand LOWER bound further
                if self.margin > 0:
                    vals_lt = nums[mask_lt]
                    min_lt_value = float(vals_lt.min())
                    
                    # Expand downward from the smallest < value
                    expansion_amount = self.margin * abs(min_lt_value) if min_lt_value != 0 else self.margin
                    proposed_min = min_lt_value - expansion_amount
                    
                    #expand if smaller
                    if proposed_min < global_min:
                        old = global_min
                        pending_new_min[col] = proposed_min

                        #log new expancion expansion
                        expansion_reason = f"due to new < values (actual < {sorted(set(vals_lt))}) in transform"
                        self._log_expansion("transform", col, "min", old, proposed_min, expansion_reason)

            # -------------------- >x values --------------------------------
            mask_gt = (types == "gt") & np.isfinite(nums)
            if mask_gt.any():
                # For >x values: min = global_min, max = global_max
                col_min[mask_gt] = global_min
                col_max[mask_gt] = global_max
                
                # Check if expand UPPER bound further
                if self.margin > 0:
                    vals_gt = nums[mask_gt]
                    max_gt_value = float(vals_gt.max())
                    
                    # Expand upward from the largest > value
                    expansion_amount = self.margin * abs(max_gt_value) if max_gt_value != 0 else self.margin
                    proposed_max = max_gt_value + expansion_amount
                    
                    #expand if bigger
                    if proposed_max > global_max:
                        old = global_max
                        pending_new_max[col] = proposed_max

                        #log new expancion expansion
                        expansion_reason = f"due to new > values (actual > {sorted(set(vals_gt))}) in transform"
                        self._log_expansion("transform", col, "max", old, proposed_max, expansion_reason)

            # collect outputs
            out[f"{col}_min"] = col_min
            out[f"{col}_max"] = col_max
            if is_nogotea_col:
                out[f"{col}_No_Gotea"] = col_NG

        # ---------------- Update stored limits if expansions occurred ----------------
        for col, new_min in pending_new_min.items():
            self.col_global_min_[col] = float(new_min)
            self.col_has_expanded_min_[col] = True

        for col, new_max in pending_new_max.items():
            self.col_global_max_[col] = float(new_max)
            self.col_has_expanded_max_[col] = True

        # ---------------- RETURN FULL DATASET ----------------
        X_dropped = X.drop(columns=cols)
        transformed = pd.DataFrame(out, index=X.index)
        final = pd.concat([X_dropped, transformed], axis=1)

        return final

    # ----------------------------------------------------------------------
    #
    #                       EXPANSION SUMMARY PUBLIC
    #
    # ----------------------------------------------------------------------
    def get_expansion_summary(self):
        """Returns a summary of all expansions that occurred"""
        if not self.expansion_log:
            return "No expansions occurred."
        
        summary = "EXPANSION SUMMARY:\n"
        summary += "=" * 80 + "\n"
        
        for i, exp in enumerate(self.expansion_log, 1):
            summary += f"{i}. Stage: {exp['stage'].upper()}, "
            summary += f"Column: {exp['column']}, "
            summary += f"Limit: {exp['limit_type'].upper()}, "
            summary += f"Old: {exp['old_value']} → New: {exp['new_value']}, "
            summary += f"Margin: {exp['margin']}, "
            summary += f"Reason: {exp['reason']}\n"
        
        return summary