// additional context as to what we're currently visiting
enum class CONTEXT {
    FUNCTION,
    DECL_BODY, // inside `type qualifier ID = ***`
    VECTOR_LITERAL, // inside a VectorNode
    NONE,
};