#include <cassert>
#include <cstdio>

#include "time_manager.hpp"

int main() {
    using zqtime::TimeControl;

    {
        const auto b = zqtime::allocate(TimeControl{180000, 2000, 0, 0, 20});
        assert(b.optimumMs > 2000);
        assert(b.optimumMs < 10000);
        assert(b.maximumMs >= b.optimumMs);
        assert(b.maximumMs < 180000);
    }

    {
        const auto noInc = zqtime::allocate(TimeControl{180000, 0, 0, 0, 20});
        const auto withInc = zqtime::allocate(TimeControl{180000, 2000, 0, 0, 20});
        assert(withInc.optimumMs > noInc.optimumMs);
    }

    {
        const auto early = zqtime::allocate(TimeControl{60000, 1000, 0, 0, 20});
        const auto late = zqtime::allocate(TimeControl{60000, 1000, 60, 0, 20});
        assert(late.optimumMs >= early.optimumMs);
    }

    {
        const auto low = zqtime::allocate(TimeControl{500, 2000, 40, 0, 20});
        assert(low.optimumMs >= 1);
        assert(low.maximumMs >= low.optimumMs);
        assert(low.maximumMs < 500);
    }

    {
        const auto explicitMtg = zqtime::allocate(TimeControl{30000, 1000, 20, 5, 20});
        assert(explicitMtg.optimumMs > 0);
        assert(explicitMtg.maximumMs < 30000);
    }

    std::puts("time_manager: OK");
    return 0;
}
