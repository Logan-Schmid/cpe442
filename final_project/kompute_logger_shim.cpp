// Fallback for Kompute static installs that reference logger::setupLogger()
// without exporting it in linked libraries.
namespace logger {
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void setupLogger() {}
} // namespace logger
