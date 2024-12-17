### Results of the Survey on Developers' Perspectives on Testing Methods for Deep Learning (DL) Libraries

#### Question 1 (Multiple Choice)
Model-level testing methods evaluate the functionality of the DL library APIs by directly executing the entire DL model. What is your opinion on model-level testing methods for DL libraries?

**Options:**
- Bugs detected during model-level testing often involve many APIs, making it difficult to trace specific erroneous APIs. This results in inefficient debugging and fixing. **[10 developers selected (100%)]**
- Model-level testing requires entire models as inputs. However, the number of available models is limited, which leads to insufficient testing. **[6 developers selected (60%)]**
- Bugs detected during model-level testing are easy to locate, and debugging/fixing is efficient. **[1 developer selected (10%)]**

**Total Responses:** 10

---

#### Question 2 (Multiple Choice)
API sequence-based testing methods for DL libraries involve testing sequences of multiple APIs (typically fewer than ten). What is your opinion on this type of testing method for DL libraries?

**Options:**
- This testing reveals interaction issues that single API tests fail to detect. **[10 developers selected (100%)]**
- In real inference or training scenarios, some API sequences are frequently used. Bugs within these sequences may affect many models. Consequently, testing frequently occurring API sequences is more meaningful than testing randomly combined API sequences. **[8 developers selected (80%)]**
- DL library APIs may exhibit precision errors when running on different hardware, and these errors can accumulate, potentially leading to non-convergence on certain hardware. Testing API sequences helps identify these accumulation trends and detect issues before the entire model fails to converge. **[8 developers selected (80%)]**
- Because the number of APIs in an API sequence is fewer than the entire model, bugs detected through the API sequence-based testing method are easier to locate, and debugging/fixing is more efficient than model-level testing. **[6 developers selected (60%)]**

**Total Responses:** 10

---

#### Question 3 (Multiple Choice)
If the inputs used to test the DL library APIs match the features (e.g., type, range, etc.) of the inputs that the APIs would encounter in real inference or training scenarios, the inputs are considered real. If the test inputs do not match the features of real inputs (e.g., matrices filled with extreme values), they are considered unrealistic. What is your opinion on the realism of test inputs?

**Options:**
- Many bugs may only be triggered under real input conditions. Focusing on testing under real input conditions enables developers to prioritize identifying and addressing issues that are likely to be encountered in actual applications of the DL library. **[10 developers selected (100%)]**
- Unrealistic inputs may only reveal bugs that are less likely to occur in real-world scenarios. **[6 developers selected (60%)]**
- To achieve comprehensive testing of the library, it is necessary to consider a variety of input scenarios, including both real and unrealistic inputs. **[7 developers selected (70%)]**

**Total Responses:** 10

