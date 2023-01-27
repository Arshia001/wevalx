#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* partial-evaluation async requests and queues                              */
/* ------------------------------------------------------------------------- */

typedef void (*weval_func_t)();

typedef struct weval_req_t weval_req_t;

struct weval_req_t {
    weval_req_t* next;
    weval_func_t func;
    uint64_t func_ctx;
    uint64_t pc_ctx;
    weval_func_t* specialized;
};

extern weval_req_t* weval_req_pending_head;
extern weval_req_t* weval_req_freelist_head;

static void weval_request(weval_req_t* req) {
    req->next = weval_req_pending_head;
    weval_req_pending_head = req;
}

static void weval_free() {
    weval_req_t* next = NULL;
    for (; weval_req_freelist_head; weval_req_freelist_head = next) {
        next = weval_req_freelist_head->next;
        if (weval_req_freelist_head->args) {
            free(weval_req_freelist_head->args);
        }
        free(weval_req_freelist_head);
    }
    weval_req_freelist_head = NULL;
}

/* ------------------------------------------------------------------------- */
/* intrinsics                                                                */
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/* "bless" a pointer so that all loads from it, directly and
 * indirectly, are "const" and allowed to see values during partial
 * evaluation. */
__attribute__((noinline))
const void* weval_assume_const_memory(const void* p);

/* Start a specialized region. Should come before any unrolled
 * loop. Returns the function context. */
__attribute__((noinline))
uint64_t weval_start(uint64_t func_ctx, uint64_t pc_ctx, void* funcptr);
/* Within a specialized region, update the PC ctx. Value returned
 * should be used for all accesses throughout the specialized
 * region. If updated, next basic block edge goes to block of new
 * context. */
__attribute__((noinline))
uint64_t weval_pc_ctx(uint64_t pc_ctx);
/* End a specialized region. Should come after any loop. */
__attribute__((noinline))
void weval_end();

/* Note a func_ctx/pc_ctx pair that may occur, with the funcptr
 * associated with it. */
__attribute__((noinline))
void weval_register(uint64_t func_ctx, uint64_t pc_ctx, void* funcptr);

/* "bless" a pointer for memory renaming. */
__attribute__((noinline))
void* weval_make_symbolic_ptr(void* p);
/* flush a region of renamed memory back to memory, returning a pointer. */
__attribute__((noinline))
void* weval_flush_to_mem(void* p, uint32_t len);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
namespace weval {
template<typename T>
const T* assume_const_memory(const T* t) {
    return (const T*)weval_assume_const_memory((const void*)t);
}

static void push_context(uint32_t pc) {
    weval_push_context(pc);
}

static void pop_context() {
    weval_pop_context();
}

static void update_context(uint32_t pc) {
    weval_update_context(pc);
}
template<typename T>
static T* make_symbolic_ptr(T* t) {
    return (T*)weval_make_symbolic_ptr((void*)t);
}
template<typename T>
void flush_to_mem(T* p, size_t len) {
    weval_flush_to_mem((void*)p, (uint32_t)len);
}

}  // namespace weval
#endif  // __cplusplus

/* ------------------------------------------------------------------------- */
/* C++ type-safe wrapper for partial evaluation of functions                 */
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
namespace weval {

template<typename T>
struct Specialize {
    typedef T value_t;
    bool is_specialized;
    T value;

    Specialize() : is_specialized(false) {}
    explicit Specialize(T value_) : is_specialized(true), value(value_) {}
};

template<typename T>
static Specialize<T> Runtime() {
    return Specialize<T>();
}

namespace impl {
template<typename Ret, typename... Args>
using FuncPtr = Ret (*)(Args...);

template<typename T>
struct StoreArg;

template<>
struct StoreArg<uint32_t> {
    void operator()(weval_req_arg_t* arg, uint32_t value) {
        arg->specialize = 1;
        arg->ty = weval_req_arg_i32;
        arg->u.i32 = value;
    }
};
template<>
struct StoreArg<uint64_t> {
    void operator()(weval_req_arg_t* arg, uint64_t value) {
        arg->specialize = 1;
        arg->ty = weval_req_arg_i64;
        arg->u.i64 = value;
    }
};
template<>
struct StoreArg<float> {
    void operator()(weval_req_arg_t* arg, float value) {
        arg->specialize = 1;
        arg->ty = weval_req_arg_f32;
        arg->u.f32 = value;
    }
};
template<>
struct StoreArg<double> {
    void operator()(weval_req_arg_t* arg, double value) {
        arg->specialize = 1;
        arg->ty = weval_req_arg_f64;
        arg->u.f64 = value;
    }
};
template<typename T>
struct StoreArg<T*> {
    void operator()(weval_req_arg_t* arg, T* value) {
        static_assert(sizeof(T*) == 4, "Only 32-bit Wasm supported");
        arg->specialize = 1;
        arg->ty = weval_req_arg_i32;
        arg->u.i32 = reinterpret_cast<uint32_t>(value);
    }
};
template<typename T>
struct StoreArg<const T*> {
    void operator()(weval_req_arg_t* arg, const T* value) {
        static_assert(sizeof(const T*) == 4, "Only 32-bit Wasm supported");
        arg->specialize = 1;
        arg->ty = weval_req_arg_i32;
        arg->u.i32 = reinterpret_cast<uint32_t>(value);
    }
};

template<typename... Args>
struct StoreArgs {};

template<>
struct StoreArgs<> {
    void operator()(weval_req_arg_t* args) {}
};

template<typename Arg, typename... Rest>
struct StoreArgs<Arg, Rest...> {
    void operator()(weval_req_arg_t* args, Arg arg0, Rest... rest) {
        if (arg0.is_specialized) {
            StoreArg<typename Arg::value_t>()(args, arg0.value);
        } else {
            args[0].specialize = 0;
        }
        StoreArgs<Rest...>()(args + 1, rest...);
    }
};

}  // impl

template<typename Ret, typename... Args, typename... WrappedArgs>
bool weval(impl::FuncPtr<Ret, Args...>* dest, impl::FuncPtr<Ret, Args...> generic, Specialize<Args>... args) {
    weval_req_t* req = (weval_req_t*)malloc(sizeof(weval_req_t));
    if (!req) {
        return false;
    }
    uint32_t nargs = sizeof...(Args);
    weval_req_arg_t* arg_storage = (weval_req_arg_t*)malloc(sizeof(weval_req_arg_t) * nargs);
    if (!arg_storage) {
        return false;
    }
    impl::StoreArgs<Specialize<Args>...>()(arg_storage, args...);

    req->func = (weval_func_t)generic;
    req->args = arg_storage;
    req->nargs = nargs;
    req->specialized = (weval_func_t*)dest;

    weval_request(req);

    return true;
}

}  // namespace weval

#endif // __cplusplus
