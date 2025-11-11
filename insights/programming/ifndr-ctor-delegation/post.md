Your code's wrong, and the compiler knows -- but won't tell you💥

Meet IFNDR (Ill‑Formed, 𝐍𝐨 𝐃𝐢𝐚𝐠𝐧𝐨𝐬𝐭𝐢𝐜 𝐑𝐞𝐪𝐮𝐢𝐫𝐞𝐝): C++ allows compilers to accept certain broken programs without telling you.

A classic example here 👉https://godbolt.org/z/PKP333e5e
```cpp
#include <cstdint>

enum class Baud {
    _9600 = 9600,
    _115200 = 115200,
    _921600 = 921600,
};

class Modem {
public:
    explicit Modem(Baud b): Modem(static_cast<int>(b)) {}
    Modem(uint32_t baud) : baud_(baud) {}
    Modem(int baud) : Modem{static_cast<Baud>(baud)} {}
private:
    uint32_t baud_;
};

int main() 
{
    // IFNDR: Ill-Formed -- No Diagnostic Required
    Modem m{Baud::_115200};
}

```

A constructor-delegation cycle -- GCC (-Wall) accepts it and you hit a stack overflow at runtime; Clang rejects it.

This looks like a stupid example, but I use delegating constructors a lot, and a new overload from a PR merge/conflict resolution can sneak in.

How to avoid it
* Pick one "primary" constructor that actually initializes the object.
* Make every other constructor delegate into that one (one-way chain).
* Prefer explicit on converting ctors to limit accidental hops.
* In CI, build with both GCC and Clang (even if your target uses only one).
* UBSan won’t catch IFNDR; run ASan on host builds to catch the stack overflow.
